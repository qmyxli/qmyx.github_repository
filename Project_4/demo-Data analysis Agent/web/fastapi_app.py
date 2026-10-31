"""FastAPI后端服务"""

import sys
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel
import numpy as np
import pandas as pd

from dotenv import load_dotenv
# 加载当前目录下 .env 文件
load_dotenv()

# 添加项目路径
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 加载 .env 环境变量（大模型密钥等），并配置日志（控制台 + logs/app.log）
try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")
except ImportError:
    pass

from agent.logging_config import setup_logging
setup_logging()

from agent.core_agent import EcommerceAgent


def _json_safe(obj):
    """递归把 NaN/Inf/NaT/Timestamp 等转换为 JSON 可序列化的值"""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    return obj


class SafeJSONResponse(JSONResponse):
    """默认 JSON 响应，自动清理 NaN/Inf/NaT"""
    def render(self, content) -> bytes:
        return json.dumps(_json_safe(content), ensure_ascii=False).encode("utf-8")


# 创建FastAPI应用
app = FastAPI(
    title="电商数据分析Agent API",
    description="基于ReAct范式的智能数据分析系统",
    version="1.0.0",
    default_response_class=SafeJSONResponse
)

# 全局Agent实例
agent = EcommerceAgent(enable_rag=True, enable_react=True)

# 请求模型
class AnalyzeRequest(BaseModel):
    dataset_name: str
    analysis_type: str
    params: Optional[Dict[str, Any]] = None

class VisualizeRequest(BaseModel):
    dataset_name: str
    chart_type: str
    params: Optional[Dict[str, Any]] = None

class ReportRequest(BaseModel):
    format: str = "html"

class DatabaseConnectRequest(BaseModel):
    connection_string: str

class DatabaseLoadRequest(BaseModel):
    table_name: str
    limit: Optional[int] = None

class SQLQueryRequest(BaseModel):
    sql: str

class AskRequest(BaseModel):
    question: str

# ============ API端点 ============

@app.get("/", response_class=HTMLResponse)
async def root():
    """返回前端页面"""
    html_path = project_root / "web" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding='utf-8'))

@app.get("/api/status")
async def get_status():
    """获取Agent状态"""
    return agent.get_status()

@app.post("/api/data/upload")
async def upload_data(file: UploadFile = File(...)):
    """上传数据文件（按扩展名区分 CSV / Excel）"""
    try:
        upload_dir = project_root / "data" / "raw"
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / file.filename

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # 根据扩展名选择读取方式
        suffix = Path(file.filename).suffix.lower()
        if suffix in ('.xlsx', '.xls'):
            df = agent.collector.collect_from_excel(str(file_path))
            dataset_name = Path(file.filename).stem
            agent.data_store[dataset_name] = df
        else:
            df = agent.load_data(str(file_path))

        return {
            "success": True,
            "message": f"数据加载成功: {len(df)} 行, {len(df.columns)} 列",
            "rows": len(df),
            "columns": len(df.columns),
            "preview": df.head(10).to_dict(orient='records')
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/data/sample")
async def load_sample_data(n_orders: int = 100):
    """加载示例数据"""
    try:
        data = agent.load_data('sample', source_type='sample', n_orders=n_orders)
        return {
            "success": True,
            "message": f"示例数据加载成功",
            "datasets": {name: len(df) for name, df in agent.data_store.items()}
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/data/datasets")
async def list_datasets():
    """列出已加载的数据集"""
    return {
        "datasets": {
            name: {
                "rows": len(df),
                "columns": list(df.columns),
                "preview": df.head(5).to_dict(orient='records')
            }
            for name, df in agent.data_store.items()
        }
    }

@app.post("/api/analyze")
async def analyze_data(request: AnalyzeRequest):
    """执行数据分析"""
    try:
        params = request.params or {}
        result = agent.analyze_data(
            request.dataset_name,
            request.analysis_type,
            **params
        )
        
        # 格式化结果
        if request.analysis_type == 'descriptive':
            return {
                "success": True,
                "type": "descriptive",
                "shape": result['shape'],
                "statistics": result['statistics'],
                "missing_values": result['missing_values']
            }
        elif request.analysis_type == 'rfm':
            return {
                "success": True,
                "type": "rfm",
                "customers": len(result),
                "segments": result['segment'].value_counts().to_dict(),
                "data": result.head(50).to_dict(orient='records')
            }
        elif request.analysis_type == 'trend':
            return {
                "success": True,
                "type": "trend",
                "data": result.to_dict(orient='records')
            }
        elif request.analysis_type == 'correlation':
            return {
                "success": True,
                "type": "correlation",
                "matrix": result.to_dict()
            }
        else:
            return {"success": True, "result": str(result)}
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/visualize")
async def visualize_data(request: VisualizeRequest):
    """生成可视化图表"""
    try:
        params = request.params or {}
        chart_path = agent.visualize_data(
            request.dataset_name,
            request.chart_type,
            **params
        )
        
        return {
            "success": True,
            "chart_path": chart_path,
            "message": f"图表生成成功: {chart_path}"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/chart/{filename}")
async def get_chart(filename: str):
    """获取图表文件"""
    chart_path = project_root / "output" / filename
    if chart_path.exists():
        return FileResponse(chart_path)
    raise HTTPException(status_code=404, detail="图表不存在")

@app.post("/api/report")
async def generate_report(request: ReportRequest):
    """生成分析报告"""
    try:
        report_path = agent.generate_report(request.format)
        
        return {
            "success": True,
            "report_path": report_path,
            "message": f"报告生成成功: {report_path}"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/report/download")
async def download_report():
    """下载报告"""
    report_path = project_root / "output" / "report.html"
    if report_path.exists():
        return FileResponse(report_path, filename="report.html")
    raise HTTPException(status_code=404, detail="报告不存在")

@app.post("/api/database/connect")
async def connect_database(request: DatabaseConnectRequest):
    """连接数据库"""
    try:
        agent.connect_database(request.connection_string)
        db_info = agent.get_database_info()
        return {
            "success": True,
            "message": "数据库连接成功",
            "info": db_info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/database/load")
async def load_from_database(request: DatabaseLoadRequest):
    """从数据库加载表"""
    try:
        df = agent.load_from_database(request.table_name, request.limit)
        return {
            "success": True,
            "message": f"表 {request.table_name} 加载成功: {len(df)} 行",
            "rows": len(df),
            "preview": df.head(10).to_dict(orient='records')
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/database/query")
async def execute_sql(request: SQLQueryRequest):
    """执行SQL查询"""
    try:
        df = agent.execute_sql(request.sql)
        return {
            "success": True,
            "rows": len(df),
            "columns": list(df.columns),
            "data": df.to_dict(orient='records')
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/ask")
async def ask_question(request: AskRequest):
    """自然语言提问（ReAct智能体）"""
    try:
        result = agent.ask(request.question)
        return {
            "success": result.get('success', True),
            "question": request.question,
            "answer": result.get('answer', ''),
            "steps": result.get('steps', []),
            "trace": result.get('trace', '')
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/knowledge/search")
async def search_knowledge(query: str):
    """搜索知识库"""
    try:
        results = agent.knowledge_base.retrieve(query, top_k=5)
        return {
            "success": True,
            "query": query,
            "results": [
                {
                    "name": r['metric_info']['name'],
                    "definition": r['metric_info']['definition'],
                    "score": r['relevance_score']
                }
                for r in results
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# 启动入口
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

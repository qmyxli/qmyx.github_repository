"""启动FastAPI Web服务并自动打开浏览器"""

import sys
import time
import threading
import webbrowser
import logging
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 加载 .env 环境变量（大模型密钥等）
try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")
except ImportError:
    pass

# 配置日志（控制台 + logs/app.log），并让 uvicorn 日志一并写入 logs/uvicorn.log
from agent.logging_config import setup_logging
log_dir = setup_logging()

uvicorn_file_handler = logging.FileHandler(log_dir / "uvicorn.log", encoding="utf-8")
uvicorn_file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logging.getLogger("uvicorn").addHandler(uvicorn_file_handler)
logging.getLogger("uvicorn.error").addHandler(uvicorn_file_handler)
logging.getLogger("uvicorn.access").addHandler(uvicorn_file_handler)


def open_browser():
    """延迟打开浏览器，等待服务启动"""
    time.sleep(2)
    webbrowser.open("http://localhost:8000")


def main():
    """启动FastAPI应用"""
    print("=" * 60)
    print("电商数据分析Agent - FastAPI服务启动中...")
    print("=" * 60)
    print("\n服务地址: http://localhost:8000")
    print("API文档: http://localhost:8000/docs")
    print("\n按 Ctrl+C 停止服务\n")

    # 在后台线程中打开浏览器
    threading.Timer(2, open_browser).start()

    # 启动FastAPI
    import uvicorn
    uvicorn.run(
        "web.fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )


if __name__ == "__main__":
    main()

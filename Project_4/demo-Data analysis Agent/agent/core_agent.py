"""核心Agent引擎"""

import sys
from pathlib import Path
import pandas as pd
import logging
from typing import Dict, List, Any, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from agent.tools.data_collector import DataCollector
from agent.tools.data_cleaner import DataCleaner
from agent.tools.data_analyzer import DataAnalyzer
from agent.tools.data_visualizer import DataVisualizer
from agent.tools.report_generator import ReportGenerator
from agent.tools.database_connector import DatabaseConnector
from agent.tools.nl2sql import NL2SQLEngine
from agent.knowledge.knowledge_base import MetricKnowledgeBase
from agent.react_agent import ReActAgent
from agent.llm_client import LLMClient
from agent.logging_config import setup_logging

logger = logging.getLogger(__name__)


class EcommerceAgent:
    """电商数据分析Agent"""
    
    def __init__(self, config_path: str = None, enable_rag: bool = True,
                 enable_react: bool = True, llm=None):
        """
        初始化Agent
        
        Args:
            config_path: 配置文件路径
            enable_rag: 是否启用RAG知识库
            enable_react: 是否启用ReAct智能体
            llm: LLM模型实例（可选）
        """
        self.config = {}
        # 未显式传入 llm 时，尝试从环境变量(.env)构建大模型客户端；
        # 未配置密钥则保持为 None，让 ReAct / NL2SQL 自动降级为内置规则逻辑
        if llm is None:
            client = LLMClient()
            llm = client if client.is_configured else None
        self.llm = llm
        if self.llm is not None:
            logger.info("✓ 大模型已接入")
        else:
            logger.info("未配置大模型 API 密钥，智能体将使用内置规则逻辑")
        self._setup_logging()
        
        # 初始化工具
        self.collector = DataCollector()
        self.cleaner = DataCleaner()
        self.visualizer = None
        self.analyzer = None
        self.db_connector = None  # 数据库连接器
        
        # 数据存储
        self.data_store: Dict[str, pd.DataFrame] = {}
        self.analysis_results: Dict[str, Any] = {}
        
        # RAG知识库
        self.knowledge_base = None
        self.nl2sql_engine = None
        if enable_rag:
            self.knowledge_base = MetricKnowledgeBase()
            logger.info("✓ RAG知识库已启用")
        
        # ReAct智能体
        self.react_agent = None
        if enable_react:
            self._init_react_agent()
            logger.info("✓ ReAct智能体已启用")
        
        logger.info("✓ 电商数据分析Agent初始化完成")
    
    def _setup_logging(self):
        """设置日志：控制台 + logs/app.log（按天滚动）"""
        setup_logging()
    
    def load_data(self, source: str, source_type: str = 'csv', **kwargs) -> pd.DataFrame:
        """加载数据"""
        logger.info(f"加载数据: {source} (类型: {source_type})")
        
        if source_type == 'sample':
            data_dict = self.collector.generate_sample_data(**kwargs)
            self.data_store.update(data_dict)
            return data_dict
        elif source_type == 'csv':
            df = self.collector.collect_from_csv(source, **kwargs)
            dataset_name = Path(source).stem
            self.data_store[dataset_name] = df
            return df
        else:
            raise ValueError(f"不支持的数据源类型: {source_type}")
    
    def clean_data(self, dataset_name: str, config: Optional[Dict] = None) -> pd.DataFrame:
        """清洗数据"""
        logger.info(f"清洗数据: {dataset_name}")
        
        if dataset_name not in self.data_store:
            raise ValueError(f"数据集不存在: {dataset_name}")
        
        df = self.data_store[dataset_name]
        df_cleaned = self.cleaner.clean_pipeline(df, config)
        
        # 更新存储
        cleaned_name = f"{dataset_name}_cleaned"
        self.data_store[cleaned_name] = df_cleaned
        
        logger.info(f"✓ 数据清洗完成: {len(df_cleaned)} 行")
        return df_cleaned
    
    def analyze_data(self, dataset_name: str, analysis_type: str, **kwargs) -> Any:
        """执行数据分析"""
        logger.info(f"执行分析: {analysis_type} on {dataset_name}")
        
        if dataset_name not in self.data_store:
            raise ValueError(f"数据集不存在: {dataset_name}")
        
        df = self.data_store[dataset_name]
        self.analyzer = DataAnalyzer(df)
        
        if analysis_type == 'descriptive':
            result = self.analyzer.descriptive_analysis()
        elif analysis_type == 'rfm':
            result = self.analyzer.rfm_analysis(
                customer_col=kwargs.get('customer_col', 'customer_id'),
                date_col=kwargs.get('date_col', 'order_date'),
                amount_col=kwargs.get('amount_col', 'total_amount')
            )
        elif analysis_type == 'trend':
            result = self.analyzer.trend_analysis(
                date_col=kwargs.get('date_col', 'order_date'),
                value_col=kwargs.get('value_col', 'total_amount'),
                freq=kwargs.get('freq', 'ME')
            )
        elif analysis_type == 'correlation':
            result = self.analyzer.correlation_analysis(
                columns=kwargs.get('columns')
            )
        else:
            raise ValueError(f"不支持的分析类型: {analysis_type}")
        
        # 存储结果
        self.analysis_results[analysis_type] = result
        
        logger.info(f"✓ 分析完成: {analysis_type}")
        return result
    
    def visualize_data(self, dataset_name: str, chart_type: str, **kwargs) -> str:
        """生成可视化图表"""
        logger.info(f"生成图表: {chart_type} on {dataset_name}")
        
        if dataset_name not in self.data_store:
            raise ValueError(f"数据集不存在: {dataset_name}")
        
        df = self.data_store[dataset_name]
        output_dir = str(project_root / "output")
        self.visualizer = DataVisualizer(df, output_dir)
        
        if chart_type == 'sales_trend':
            path = self.visualizer.plot_sales_trend(
                date_col=kwargs.get('date_col', 'order_date'),
                value_col=kwargs.get('value_col', 'total_amount')
            )
        elif chart_type == 'customer_segments':
            if 'rfm' not in self.analysis_results:
                raise ValueError("请先执行RFM分析")
            rfm_df = self.analysis_results['rfm']
            path = self.visualizer.plot_customer_segments(rfm_df)
        elif chart_type == 'correlation_heatmap':
            path = self.visualizer.plot_correlation_heatmap(
                columns=kwargs.get('columns')
            )
        else:
            raise ValueError(f"不支持的图表类型: {chart_type}")
        
        logger.info(f"✓ 图表生成完成: {path}")
        return path
    
    def generate_report(self, report_format: str = 'html', **kwargs) -> str:
        """生成分析报告"""
        logger.info(f"生成报告: {report_format}")

        output_dir = str(project_root / "output")
        
        # 整合分析结果
        full_results = {
            'summary': {
                'total_records': sum(len(df) for df in self.data_store.values()),
                'datasets': list(self.data_store.keys()),
                'analyses_performed': list(self.analysis_results.keys())
            },
            'results': self.analysis_results,
            'insights': self._generate_insights()
        }
        
        reporter = ReportGenerator(full_results, output_dir)
        
        if report_format == 'markdown':
            path = reporter.generate_markdown_report()
        elif report_format == 'html':
            path = reporter.generate_html_report()
        else:
            raise ValueError(f"不支持的报告格式: {report_format}")
        
        logger.info(f"✓ 报告生成完成: {path}")
        return path
    
    def _generate_insights(self) -> List[str]:
        """生成业务洞察"""
        insights = []
        
        if 'rfm' in self.analysis_results:
            rfm = self.analysis_results['rfm']
            high_value_pct = (rfm['segment'] == '高价值客户').sum() / len(rfm) * 100
            insights.append(f"高价值客户占比: {high_value_pct:.1f}%")
            
            churn_risk_pct = (rfm['segment'] == '流失风险客户').sum() / len(rfm) * 100
            insights.append(f"流失风险客户占比: {churn_risk_pct:.1f}%")
        
        if 'trend' in self.analysis_results:
            trend = self.analysis_results['trend']
            if 'growth_rate' in trend:
                avg_growth = trend['growth_rate'].mean()
                if not pd.isna(avg_growth):
                    insights.append(f"平均增长率: {avg_growth:.2f}%")
        
        return insights
    
    def run_full_pipeline(self, data_source: str = None, source_type: str = 'sample', **kwargs) -> Dict[str, Any]:
        """运行完整分析流程"""
        logger.info("=" * 60)
        logger.info("开始完整分析流程")
        logger.info("=" * 60)
        
        try:
            # 1. 加载数据
            logger.info("\n[1/5] 加载数据...")
            if data_source:
                df = self.load_data(data_source, source_type, **kwargs)
                dataset_name = Path(data_source).stem
            else:
                data = self.load_data('sample', source_type='sample', n_orders=100)
                dataset_name = 'orders'
                df = data
            
            # 2. 清洗数据
            logger.info("\n[2/5] 清洗数据...")
            df_cleaned = self.clean_data(dataset_name)
            cleaned_name = f"{dataset_name}_cleaned"
            
            # 3. 执行分析
            logger.info("\n[3/5] 执行分析...")
            self.analyze_data(cleaned_name, 'descriptive')
            
            try:
                self.analyze_data(
                    cleaned_name, 'rfm',
                    customer_col='customer_id',
                    date_col='order_date',
                    amount_col='total_amount'
                )
            except Exception as e:
                logger.warning(f"RFM分析失败: {e}")
            
            try:
                self.analyze_data(
                    cleaned_name, 'trend',
                    date_col='order_date',
                    value_col='total_amount'
                )
            except Exception as e:
                logger.warning(f"趋势分析失败: {e}")
            
            # 4. 生成可视化
            logger.info("\n[4/5] 生成可视化...")
            try:
                self.visualize_data(cleaned_name, 'sales_trend')
            except Exception as e:
                logger.warning(f"销售趋势图生成失败: {e}")
            
            if 'rfm' in self.analysis_results:
                try:
                    self.visualize_data(cleaned_name, 'customer_segments')
                except Exception as e:
                    logger.warning(f"客户细分图生成失败: {e}")
            
            # 5. 生成报告
            logger.info("\n[5/5] 生成报告...")
            report_path = self.generate_report('html')
            
            logger.info("\n" + "=" * 60)
            logger.info("✓ 完整分析流程完成")
            logger.info("=" * 60)
            
            return {
                'status': 'success',
                'data_loaded': len(df),
                'data_cleaned': len(df_cleaned),
                'analyses': list(self.analysis_results.keys()),
                'report_path': report_path
            }
            
        except Exception as e:
            logger.error(f"分析流程失败: {e}", exc_info=True)
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def connect_database(self, connection_string: str) -> bool:
        """连接到SQL数据库"""
        logger.info(f"连接数据库: {connection_string[:50]}...")
        
        try:
            if not self.db_connector:
                self.db_connector = DatabaseConnector(connection_string)
            else:
                self.db_connector.connect(connection_string)
            
            # 初始化NL2SQL引擎（需要数据库+知识库）
            if self.knowledge_base:
                self.nl2sql_engine = NL2SQLEngine(
                    db_connector=self.db_connector,
                    knowledge_base=self.knowledge_base,
                    llm=self.llm,
                    max_retries=3
                )
                logger.info("✓ NL2SQL引擎已初始化")
            
            logger.info("✓ 数据库连接成功")
            return True
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            raise
    
    def load_from_database(self, table_name: str, limit: Optional[int] = None) -> pd.DataFrame:
        """从数据库加载表数据"""
        logger.info(f"从数据库加载表: {table_name}")
        
        if not self.db_connector:
            raise ValueError("未连接数据库，请先调用connect_database()")
        
        df = self.db_connector.load_table(table_name, limit)
        self.data_store[table_name] = df
        
        logger.info(f"✓ 加载表 {table_name}: {len(df)} 行")
        return df
    
    def execute_sql(self, sql_query: str) -> pd.DataFrame:
        """执行SQL查询"""
        logger.info(f"执行SQL查询: {sql_query[:100]}...")
        
        if not self.db_connector:
            raise ValueError("未连接数据库")
        
        df = self.db_connector.execute_query(sql_query)
        logger.info(f"✓ SQL查询完成: {len(df)} 行")
        return df
    
    def get_database_info(self) -> Dict[str, Any]:
        """获取数据库信息"""
        if not self.db_connector:
            return {"status": "未连接"}
        
        return self.db_connector.get_database_info()
    
    def _init_react_agent(self):
        """初始化ReAct智能体"""
        # 定义ReAct可用工具
        tools = {
            'nl2sql_query': self._react_nl2sql_query,
            'analyze_data': self._react_analyze_data,
            'generate_chart': self._react_generate_chart,
            'generate_report': self._react_generate_report,
            'search_knowledge': self._react_search_knowledge
        }
        
        self.react_agent = ReActAgent(
            tools=tools,
            llm=self.llm,
            max_steps=10
        )
    
    def _react_nl2sql_query(self, query: str) -> str:
        """ReAct工具: NL2SQL查询"""
        if not self.nl2sql_engine:
            return "NL2SQL引擎未初始化，请先连接数据库"
        
        result = self.nl2sql_engine.natural_language_to_sql(query)
        if result['validated']:
            return f"查询成功:\n{result['result'].to_string()}"
        else:
            return f"查询失败: {result['explanation']}"
    
    def _react_analyze_data(self, analysis_type: str = 'descriptive', **kwargs) -> str:
        """ReAct工具: 数据分析"""
        if not self.data_store:
            return "没有可用数据集，请先加载数据"
        
        dataset_name = list(self.data_store.keys())[0]
        try:
            result = self.analyze_data(dataset_name, analysis_type, **kwargs)
            return f"分析完成: {analysis_type}"
        except Exception as e:
            return f"分析失败: {e}"
    
    def _react_generate_chart(self, chart_type: str = 'sales_trend', **kwargs) -> str:
        """ReAct工具: 生成图表"""
        if not self.data_store:
            return "没有可用数据集"
        
        dataset_name = list(self.data_store.keys())[0]
        try:
            path = self.visualize_data(dataset_name, chart_type, **kwargs)
            return f"图表已生成: {path}"
        except Exception as e:
            return f"图表生成失败: {e}"
    
    def _react_generate_report(self, format: str = 'html') -> str:
        """ReAct工具: 生成报告"""
        try:
            path = self.generate_report(format)
            return f"报告已生成: {path}"
        except Exception as e:
            return f"报告生成失败: {e}"
    
    def _react_search_knowledge(self, query: str) -> str:
        """ReAct工具: 知识库检索"""
        if not self.knowledge_base:
            return "知识库未启用"
        
        results = self.knowledge_base.retrieve(query, top_k=3)
        if results:
            output = []
            for r in results:
                metric = r['metric_info']
                output.append(f"- {metric['name']}: {metric['definition']}")
            return "\n".join(output)
        else:
            return "未找到相关指标"
    
    def ask(self, question: str) -> Dict[str, Any]:
        """
        自然语言问答接口（端到端自动化）

        Args:
            question: 自然语言问题

        Returns:
            结构化答案字典
        """
        if not self.react_agent:
            return {
                'success': False,
                'answer': 'ReAct智能体未启用',
                'steps': [],
                'trace': ''
            }

        result = self.react_agent.run(question)
        steps = [
            {
                'step_num': s.step_num,
                'thought': s.thought,
                'action': s.action,
                'action_input': s.action_input,
                'observation': s.observation,
                'is_final': s.is_final
            }
            for s in result.steps
        ]
        return {
            'success': result.success,
            'answer': result.final_answer,
            'steps': steps,
            'trace': self.react_agent.format_trace()
        }
    
    def get_status(self) -> Dict[str, Any]:
        """获取Agent状态"""
        return {
            'datasets_loaded': list(self.data_store.keys()),
            'analyses_completed': list(self.analysis_results.keys()),
            'database_connected': self.db_connector is not None,
            'rag_enabled': self.knowledge_base is not None,
            'react_enabled': self.react_agent is not None
        }


if __name__ == "__main__":
    # 测试用例：初始化Agent、加载示例数据、执行描述性分析
    logging.disable(logging.INFO)
    agent = EcommerceAgent()
    agent.load_data('sample', source_type='sample', n_orders=50)
    result = agent.analyze_data('orders', 'descriptive')
    assert result['shape'][0] == 50, "描述性分析结果行数不符"
    status = agent.get_status()
    assert 'orders' in status['datasets_loaded'], "数据集未加载"
    print(f"测试通过: Agent分析完成，数据集 {status['datasets_loaded']}")

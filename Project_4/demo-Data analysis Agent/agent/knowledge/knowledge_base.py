"""指标口径RAG知识库 - 统一业务统计标准，约束模型输出"""

import json
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MetricKnowledgeBase:
    """指标口径知识库 - 存储业务指标定义、计算口径、使用场景"""
    
    def __init__(self, knowledge_file: Optional[str] = None):
        """
        初始化知识库
        
        Args:
            knowledge_file: 知识库JSON文件路径，None则使用默认指标
        """
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.business_rules: List[Dict[str, Any]] = []
        
        if knowledge_file and Path(knowledge_file).exists():
            self._load_from_file(knowledge_file)
        else:
            self._init_default_metrics()
        
        logger.info(f"✓ 知识库初始化完成: {len(self.metrics)} 个指标, "
                   f"{len(self.business_rules)} 条业务规则")
    
    def _init_default_metrics(self):
        """初始化默认电商指标体系"""
        # 核心业务指标
        self.metrics = {
            "GMV": {
                "name": "GMV（成交总额）",
                "definition": "所有已完成订单的总金额，不含退款和取消订单",
                "formula": "SUM(total_amount) WHERE status = 'completed'",
                "sql_template": "SELECT SUM(total_amount) as gmv FROM orders WHERE status = 'completed'",
                "keywords": ["gmv", "总销售额", "销售额", "成交额", "卖了多少", "卖了多少钱", "收入", "营收", "成交"],
                "unit": "元",
                "data_source": "orders",
                "filters": {"status": "completed"},
                "aggregation": "SUM",
                "field": "total_amount",
                "related_metrics": ["客单价", "订单数", "支付转化率"],
                "business_meaning": "衡量平台交易规模的核心指标"
            },
            "订单数": {
                "name": "订单数",
                "definition": "指定时间范围内的订单总量",
                "formula": "COUNT(order_id)",
                "sql_template": "SELECT COUNT(order_id) as order_count FROM orders",
                "keywords": ["订单数", "订单量", "多少单", "有多少订单", "单量"],
                "unit": "单",
                "data_source": "orders",
                "filters": {},
                "aggregation": "COUNT",
                "field": "order_id",
                "related_metrics": ["GMV", "客单价", "取消率"],
                "business_meaning": "反映平台交易活跃度"
            },
            "客单价": {
                "name": "客单价（AOV）",
                "definition": "平均每笔订单的金额",
                "formula": "GMV / 订单数",
                "sql_template": "SELECT AVG(total_amount) as aov FROM orders WHERE status = 'completed'",
                "keywords": ["客单价", "aov", "平均订单金额", "平均每单", "每单平均"],
                "unit": "元/单",
                "data_source": "orders",
                "filters": {"status": "completed"},
                "aggregation": "AVG",
                "field": "total_amount",
                "related_metrics": ["GMV", "订单数"],
                "business_meaning": "衡量客户消费能力和购买意愿"
            },
            "取消率": {
                "name": "订单取消率",
                "definition": "取消订单数占总订单数的比例",
                "formula": "COUNT(status='cancelled') / COUNT(*) * 100%",
                "sql_template": "SELECT COUNT(CASE WHEN status = 'cancelled' THEN 1 END) * 100.0 / COUNT(*) as cancel_rate FROM orders",
                "keywords": ["取消率", "取消", "取消了多少"],
                "unit": "%",
                "data_source": "orders",
                "filters": {},
                "aggregation": "CUSTOM",
                "field": "status",
                "related_metrics": ["订单数", "退款率"],
                "business_meaning": "反映订单质量和用户体验"
            },
            "退款率": {
                "name": "退款率",
                "definition": "退款订单数占总订单数的比例",
                "formula": "COUNT(status='refunded') / COUNT(*) * 100%",
                "sql_template": "SELECT COUNT(CASE WHEN status = 'refunded' THEN 1 END) * 100.0 / COUNT(*) as refund_rate FROM orders",
                "keywords": ["退款率", "退款", "退了多少"],
                "unit": "%",
                "data_source": "orders",
                "filters": {},
                "aggregation": "CUSTOM",
                "field": "status",
                "related_metrics": ["订单数", "取消率", "客户满意度"],
                "business_meaning": "衡量商品质量和客户满意度"
            },
            "活跃客户数": {
                "name": "活跃客户数",
                "definition": "指定时间范围内有消费行为的客户数量",
                "formula": "COUNT(DISTINCT customer_id)",
                "sql_template": "SELECT COUNT(DISTINCT customer_id) as active_customers FROM orders",
                "keywords": ["活跃客户数", "客户数", "有多少客户", "客户数量"],
                "unit": "人",
                "data_source": "orders",
                "filters": {},
                "aggregation": "COUNT_DISTINCT",
                "field": "customer_id",
                "related_metrics": ["客户留存率", "复购率"],
                "business_meaning": "反映平台用户活跃度"
            },
            "复购率": {
                "name": "客户复购率",
                "definition": "消费次数大于1的客户占总消费客户的比例",
                "formula": "COUNT(customer_id WHERE frequency > 1) / COUNT(DISTINCT customer_id) * 100%",
                "sql_template": """
                    SELECT 
                        COUNT(CASE WHEN order_count > 1 THEN 1 END) * 100.0 / COUNT(*) as repurchase_rate
                    FROM (
                        SELECT customer_id, COUNT(*) as order_count 
                        FROM orders 
                        GROUP BY customer_id
                    ) t
                """,
                "unit": "%",
                "data_source": "orders",
                "filters": {},
                "aggregation": "CUSTOM",
                "field": "customer_id",
                "related_metrics": ["客户留存率", "RFM分析"],
                "business_meaning": "衡量客户忠诚度和平台粘性"
            },
            "高价值客户占比": {
                "name": "高价值客户占比",
                "definition": "RFM评分>=13的客户占总客户的比例",
                "formula": "COUNT(RFM_score >= 13) / COUNT(*) * 100%",
                "sql_template": "需要先执行RFM分析",
                "unit": "%",
                "data_source": "rfm_analysis",
                "filters": {"RFM_score": ">= 13"},
                "aggregation": "CUSTOM",
                "field": "RFM_score",
                "related_metrics": ["RFM分析", "客户细分"],
                "business_meaning": "衡量高质量客户群体规模"
            }
        }
        
        # 业务规则
        self.business_rules = [
            {
                "rule_id": "R001",
                "name": "订单状态过滤规则",
                "description": "计算GMV时只统计已完成订单，排除取消和退款",
                "condition": "status = 'completed'",
                "applied_metrics": ["GMV", "客单价"]
            },
            {
                "rule_id": "R002",
                "name": "时间范围默认值",
                "description": "未指定时间范围时，默认统计全部数据",
                "condition": "无时间过滤",
                "applied_metrics": ["所有指标"]
            },
            {
                "rule_id": "R003",
                "name": "客户去重规则",
                "description": "统计客户数时使用DISTINCT去重",
                "condition": "COUNT(DISTINCT customer_id)",
                "applied_metrics": ["活跃客户数", "复购率"]
            },
            {
                "rule_id": "R004",
                "name": "百分比计算规则",
                "description": "计算比率类指标时乘以100并保留2位小数",
                "condition": "ROUND(value * 100, 2)",
                "applied_metrics": ["取消率", "退款率", "复购率"]
            }
        ]
    
    def _load_from_file(self, file_path: str):
        """从JSON文件加载知识库"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.metrics = data.get('metrics', {})
            self.business_rules = data.get('business_rules', [])
            
            logger.info(f"✓ 从文件加载知识库: {file_path}")
        except Exception as e:
            logger.error(f"加载知识库失败: {e}")
            self._init_default_metrics()
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        检索相关指标（简化版RAG）
        
        Args:
            query: 用户查询文本
            top_k: 返回最相关的K个指标
        
        Returns:
            相关指标列表
        """
        query_lower = query.lower()

        # 简单的关键词匹配检索
        scored_metrics = []
        for metric_key, metric_info in self.metrics.items():
            score = 0

            # 指标名称匹配
            if metric_key.lower() in query_lower or metric_info['name'].lower() in query_lower:
                score += 10

            # 关键词匹配（keywords 字段显式列举的别名/同义表达）
            for kw in metric_info.get('keywords', []):
                if kw.lower() in query_lower:
                    score += 6

            # 定义匹配
            for keyword in query_lower.split():
                if keyword in metric_info['definition'].lower():
                    score += 3
                if keyword in metric_info.get('business_meaning', '').lower():
                    score += 2

            # 相关指标匹配
            for related in metric_info.get('related_metrics', []):
                if related.lower() in query_lower:
                    score += 1

            if score > 0:
                scored_metrics.append((score, metric_key, metric_info))
        
        # 按得分排序
        scored_metrics.sort(key=lambda x: x[0], reverse=True)
        
        # 返回top_k
        results = [
            {
                'metric_key': key,
                'metric_info': info,
                'relevance_score': score
            }
            for score, key, info in scored_metrics[:top_k]
        ]
        
        logger.info(f"检索到 {len(results)} 个相关指标: {[r['metric_key'] for r in results]}")
        return results
    
    def get_metric_definition(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """获取指标定义"""
        # 尝试精确匹配
        if metric_name in self.metrics:
            return self.metrics[metric_name]
        
        # 尝试模糊匹配
        metric_lower = metric_name.lower()
        for key, info in self.metrics.items():
            if metric_lower in key.lower() or metric_lower in info['name'].lower():
                return info
        
        return None
    
    def get_sql_template(self, metric_name: str) -> Optional[str]:
        """获取指标的SQL模板"""
        metric = self.get_metric_definition(metric_name)
        if metric:
            return metric.get('sql_template')
        return None
    
    def get_business_rules(self, metric_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取业务规则"""
        if metric_name:
            # 返回与指定指标相关的规则
            return [
                rule for rule in self.business_rules
                if metric_name in rule.get('applied_metrics', [])
            ]
        return self.business_rules
    
    def generate_prompt_context(self, query: str) -> str:
        """
        生成RAG上下文，用于约束模型输出
        
        Args:
            query: 用户查询
        
        Returns:
            格式化的上下文文本
        """
        retrieved = self.retrieve(query, top_k=3)
        
        if not retrieved:
            return "未找到相关指标定义，请明确查询的指标名称。"
        
        context_parts = ["## 相关指标定义与业务规则\n"]
        
        for i, item in enumerate(retrieved, 1):
            metric = item['metric_info']
            context_parts.append(f"### {i}. {metric['name']}")
            context_parts.append(f"- **定义**: {metric['definition']}")
            context_parts.append(f"- **计算公式**: {metric['formula']}")
            context_parts.append(f"- **单位**: {metric['unit']}")
            context_parts.append(f"- **数据来源**: {metric['data_source']}")
            
            if metric.get('filters'):
                filters_str = ", ".join([f"{k}={v}" for k, v in metric['filters'].items()])
                context_parts.append(f"- **过滤条件**: {filters_str}")
            
            context_parts.append(f"- **业务含义**: {metric.get('business_meaning', 'N/A')}")
            context_parts.append("")
        
        # 添加相关业务规则
        relevant_rules = []
        for item in retrieved:
            rules = self.get_business_rules(item['metric_key'])
            relevant_rules.extend(rules)
        
        if relevant_rules:
            context_parts.append("## 适用业务规则\n")
            for rule in relevant_rules:
                context_parts.append(f"- **{rule['name']}**: {rule['description']}")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def add_metric(self, metric_key: str, metric_info: Dict[str, Any]):
        """添加新指标"""
        self.metrics[metric_key] = metric_info
        logger.info(f"✓ 添加指标: {metric_key}")
    
    def add_business_rule(self, rule: Dict[str, Any]):
        """添加业务规则"""
        self.business_rules.append(rule)
        logger.info(f"✓ 添加业务规则: {rule.get('name', 'N/A')}")
    
    def export_to_file(self, file_path: str):
        """导出知识库到JSON文件"""
        data = {
            'metrics': self.metrics,
            'business_rules': self.business_rules
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✓ 知识库已导出: {file_path}")
    
    def list_all_metrics(self) -> List[str]:
        """列出所有指标名称"""
        return list(self.metrics.keys())
    
    def get_metric_summary(self) -> Dict[str, Any]:
        """获取知识库摘要"""
        return {
            'total_metrics': len(self.metrics),
            'total_rules': len(self.business_rules),
            'metric_names': list(self.metrics.keys()),
            'data_sources': list(set(m['data_source'] for m in self.metrics.values()))
        }


if __name__ == "__main__":
    # 测试用例：检索"收入"能命中GMV指标
    kb = MetricKnowledgeBase()
    results = kb.retrieve("公司的收入情况", top_k=1)
    assert results and results[0]['metric_key'] == 'GMV', "关键词检索未命中GMV"
    print(f"测试通过: 命中指标 {results[0]['metric_key']} (得分 {results[0]['relevance_score']})")

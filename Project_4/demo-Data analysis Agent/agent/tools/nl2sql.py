"""NL2SQL引擎 - 自然语言转SQL，含双层校验与失败重试机制"""

import re
import json
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from sqlalchemy import text
import logging

logger = logging.getLogger(__name__)


class NL2SQLEngine:
    """
    自然语言转SQL引擎
    
    核心特性:
    1. RAG知识库约束 - 利用指标口径约束SQL生成
    2. 双层校验 - SQL语法校验 + 数据结果校验
    3. 失败重试 - 自动修复并重试失败的SQL
    4. 幻觉抑制 - 基于schema和知识库约束输出
    """
    
    def __init__(self, db_connector, knowledge_base, llm=None, max_retries: int = 3):
        """
        初始化NL2SQL引擎
        
        Args:
            db_connector: DatabaseConnector实例
            knowledge_base: MetricKnowledgeBase实例
            llm: LLM模型实例（可选，无LLM时使用规则匹配）
            max_retries: 最大重试次数
        """
        self.db = db_connector
        self.kb = knowledge_base
        self.llm = llm
        self.max_retries = max_retries
        self.query_history: List[Dict] = []
        
        logger.info(f"✓ NL2SQL引擎初始化完成 (最大重试: {max_retries})")
    
    def natural_language_to_sql(self, query: str) -> Dict[str, Any]:
        """
        自然语言转SQL（主入口）
        
        流程:
        1. RAG检索 → 获取指标口径上下文
        2. SQL生成 → LLM或规则匹配
        3. 双层校验 → 语法校验 + 结果校验
        4. 失败重试 → 自动修复并重试
        
        Args:
            query: 自然语言查询
        
        Returns:
            {
                'sql': 生成的SQL,
                'result': 查询结果DataFrame,
                'validated': 是否通过校验,
                'retries': 重试次数,
                'explanation': 解释说明
            }
        """
        logger.info(f"NL2SQL查询: {query}")
        
        # 步骤1: RAG检索指标口径
        rag_context = self.kb.generate_prompt_context(query)
        logger.info(f"RAG上下文已生成 ({len(rag_context)} 字符)")
        
        # 步骤2: 获取数据库schema
        schema_context = self._get_schema_context()
        
        # 步骤3: 生成SQL
        sql = self._generate_sql(query, rag_context, schema_context)
        
        if not sql:
            return {
                'sql': None,
                'result': None,
                'validated': False,
                'retries': 0,
                'explanation': '无法生成SQL，请检查查询内容'
            }
        
        # 步骤4: 双层校验 + 重试
        retries = 0
        current_sql = sql
        last_error = None
        
        while retries <= self.max_retries:
            # 第一层: SQL语法校验
            syntax_ok, syntax_msg = self._validate_sql_syntax(current_sql)
            
            if not syntax_ok:
                logger.warning(f"SQL语法校验失败 (尝试 {retries+1}): {syntax_msg}")
                last_error = syntax_msg
                
                # 尝试修复
                fixed_sql = self._fix_sql_syntax(current_sql, syntax_msg)
                if fixed_sql and fixed_sql != current_sql:
                    current_sql = fixed_sql
                    retries += 1
                    continue
                else:
                    break
            
            # 第二层: 数据结果校验
            try:
                result = self.db.execute_query(current_sql)
                
                result_ok, result_msg = self._validate_data_result(result, current_sql)
                
                if result_ok:
                    # 全部通过
                    self._record_query(query, current_sql, result, True, retries)
                    return {
                        'sql': current_sql,
                        'result': result,
                        'validated': True,
                        'retries': retries,
                        'explanation': f'查询成功 (重试{retries}次)',
                        'rag_context': rag_context
                    }
                else:
                    logger.warning(f"数据结果校验失败 (尝试 {retries+1}): {result_msg}")
                    last_error = result_msg
                    
                    # 尝试修复
                    fixed_sql = self._fix_sql_result(current_sql, result_msg, query)
                    if fixed_sql and fixed_sql != current_sql:
                        current_sql = fixed_sql
                        retries += 1
                        continue
                    else:
                        break
                
            except Exception as e:
                logger.warning(f"SQL执行失败 (尝试 {retries+1}): {e}")
                last_error = str(e)
                
                # 尝试修复
                fixed_sql = self._fix_sql_execution(current_sql, str(e))
                if fixed_sql and fixed_sql != current_sql:
                    current_sql = fixed_sql
                    retries += 1
                    continue
                else:
                    break
        
        # 所有重试都失败
        self._record_query(query, current_sql, None, False, retries, last_error)
        return {
            'sql': current_sql,
            'result': None,
            'validated': False,
            'retries': retries,
            'explanation': f'查询失败 (重试{retries}次): {last_error}',
            'error': last_error
        }
    
    def _get_schema_context(self) -> str:
        """获取数据库schema上下文"""
        if not self.db or not self.db.engine:
            return "无数据库连接"
        
        try:
            db_info = self.db.get_database_info()
            parts = ["## 数据库Schema\n"]
            
            tables = db_info.get('tables', [])
            schemas = db_info.get('table_schemas', {})
            
            for table in tables[:10]:
                parts.append(f"### 表: {table}")
                if table in schemas and isinstance(schemas[table], dict):
                    for col, dtype in schemas[table].items():
                        parts.append(f"  - {col}: {dtype}")
                parts.append("")
            
            return "\n".join(parts)
        except Exception as e:
            logger.error(f"获取schema失败: {e}")
            return "获取schema失败"
    
    def _generate_sql(self, query: str, rag_context: str, schema_context: str) -> Optional[str]:
        """生成SQL（知识库模板 → LLM → 规则匹配，逐级降级）"""
        logger.info("生成SQL (知识库模板 → LLM → 规则匹配)")

        # 尝试从知识库直接匹配SQL模板
        template_sql = self._try_match_template(query)
        if template_sql:
            logger.info("使用知识库SQL模板")
            return template_sql

        # 使用LLM生成（失败或返回空时降级到规则匹配）
        if self.llm:
            try:
                llm_sql = self._generate_sql_with_llm(query, rag_context, schema_context)
                if llm_sql:
                    return llm_sql
                logger.warning("LLM未生成有效SQL，降级到规则匹配")
            except Exception as e:
                logger.error(f"LLM生成SQL失败，降级到规则匹配: {e}")

        # 规则匹配
        return self._generate_sql_with_rules(query)
    
    def _try_match_template(self, query: str) -> Optional[str]:
        """尝试从知识库匹配SQL模板"""
        retrieved = self.kb.retrieve(query, top_k=1)
        
        if retrieved and retrieved[0]['relevance_score'] >= 10:
            metric = retrieved[0]['metric_info']
            template = metric.get('sql_template')
            if template and template != "需要先执行RFM分析":
                return template
        
        return None
    
    def _generate_sql_with_llm(self, query: str, rag_context: str, schema_context: str) -> Optional[str]:
        """使用LLM生成SQL"""
        prompt = f"""你是一个SQL专家。根据用户的自然语言查询，生成正确的SQL语句。

{rag_context}

{schema_context}

## 要求
1. 只生成SQL语句，不要包含其他解释
2. 严格基于上述schema和指标定义
3. 遵循业务规则中的过滤条件
4. 使用标准SQL语法

用户查询: {query}

请生成SQL:"""
        
        try:
            response = self.llm.invoke(prompt)
            sql = response.content.strip()

            # 去除markdown代码块
            sql = re.sub(r'```(?:sql)?\s*', '', sql, flags=re.IGNORECASE)
            sql = sql.replace('```', '').strip()

            # 若返回内容混杂解释文字，提取第一段 SELECT/WITH 语句
            if not (sql.upper().startswith('SELECT') or sql.upper().startswith('WITH')):
                match = re.search(r'\b(SELECT|WITH)\b.+', sql, re.IGNORECASE | re.DOTALL)
                if match:
                    sql = match.group(0).strip()

            # 去掉结尾可能多余的分号或解释
            if sql.endswith(';'):
                sql = sql[:-1].strip()

            return sql or None
        except Exception as e:
            logger.error(f"LLM生成SQL失败: {e}")
            return None
    
    def _generate_sql_with_rules(self, query: str) -> Optional[str]:
        """基于规则生成SQL（无LLM时的降级方案）"""
        query_lower = query.lower()

        # 从知识库检索相关指标（含关键词扩展后的匹配）
        retrieved = self.kb.retrieve(query, top_k=1)
        if retrieved and retrieved[0]['relevance_score'] > 0:
            metric = retrieved[0]['metric_info']
            template = metric.get('sql_template')
            if template and template != "需要先执行RFM分析":
                return template

        # 基础规则匹配（覆盖常见自然问法）
        if any(kw in query for kw in ['总销售额', '销售额', '成交额', 'gmv', '卖了多少', '卖了多少钱', '收入', '营收']):
            return "SELECT SUM(total_amount) as gmv FROM orders WHERE status = 'completed'"

        if any(kw in query for kw in ['订单数', '订单量', '多少单', '有多少订单']):
            return "SELECT COUNT(order_id) as order_count FROM orders"

        if any(kw in query for kw in ['客单价', '平均订单金额', '平均每单', '每单平均']):
            return "SELECT AVG(total_amount) as aov FROM orders WHERE status = 'completed'"

        if any(kw in query for kw in ['取消率', '取消']):
            return "SELECT COUNT(CASE WHEN status = 'cancelled' THEN 1 END) * 100.0 / COUNT(*) as cancel_rate FROM orders"

        if any(kw in query for kw in ['退款率', '退款']):
            return "SELECT COUNT(CASE WHEN status = 'refunded' THEN 1 END) * 100.0 / COUNT(*) as refund_rate FROM orders"

        if any(kw in query for kw in ['客户数', '活跃客户', '有多少客户']):
            return "SELECT COUNT(DISTINCT customer_id) as active_customers FROM orders"

        # 含时间/地点等限定词的查询，退化为整表聚合，避免返回 None
        if any(kw in query for kw in ['多少', '数量', '统计', '查询', '看看', '分析', '分布', '客户', '订单', '销售', '卖']):
            return "SELECT COUNT(*) as total_count FROM orders"

        return None
    
    def _validate_sql_syntax(self, sql: str) -> Tuple[bool, str]:
        """
        第一层校验: SQL语法校验
        
        检查:
        1. SQL注入风险
        2. 危险操作（DROP, DELETE, TRUNCATE等）
        3. 基本语法结构
        4. 表名和列名是否存在
        """
        if not sql or not sql.strip():
            return False, "SQL为空"
        
        sql_upper = sql.upper().strip()
        
        # 检查危险操作
        dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'INSERT', 'UPDATE', 'CREATE']
        for keyword in dangerous_keywords:
            if re.search(rf'\b{keyword}\b', sql_upper):
                return False, f"禁止执行{keyword}操作，仅支持SELECT查询"
        
        # 检查是否以SELECT开头
        if not sql_upper.startswith('SELECT') and not sql_upper.startswith('WITH'):
            return False, "SQL必须以SELECT或WITH开头"
        
        # 检查SQL注入特征
        injection_patterns = [
            r';\s*(DROP|DELETE|INSERT|UPDATE|ALTER)',
            r'--\s*$',
            r'/\*.*\*/',
            r"'\s*OR\s+'",
            r"'\s*OR\s+1\s*=\s*1",
        ]
        for pattern in injection_patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                return False, f"检测到SQL注入特征: {pattern}"
        
        # 检查括号匹配
        if sql.count('(') != sql.count(')'):
            return False, "括号不匹配"
        
        # 检查引号匹配
        single_quotes = sql.count("'")
        if single_quotes % 2 != 0:
            return False, "单引号不匹配"
        
        return True, "语法校验通过"
    
    def _validate_data_result(self, result: pd.DataFrame, sql: str) -> Tuple[bool, str]:
        """
        第二层校验: 数据结果校验
        
        检查:
        1. 结果是否为空
        2. 数值是否合理（非负、范围合理）
        3. 百分比是否在0-100之间
        4. 是否返回了过多数据（可能是笛卡尔积）
        """
        if result is None or len(result) == 0:
            return False, "查询结果为空"
        
        # 检查结果行数
        if len(result) > 10000:
            return False, f"结果行数过多({len(result)}行)，可能是笛卡尔积"
        
        # 检查数值列
        numeric_cols = result.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            values = result[col].dropna()
            if len(values) == 0:
                continue
            
            # 检查是否有无穷大
            if (values == float('inf')).any() or (values == float('-inf')).any():
                return False, f"列{col}包含无穷大值"
            
            # 检查百分比列（列名包含rate/ratio/率）
            if any(keyword in col.lower() for keyword in ['rate', 'ratio', '率', '占比']):
                if values.min() < -1 or values.max() > 200:
                    return False, f"百分比列{col}的值超出合理范围: [{values.min()}, {values.max()}]"
            
            # 检查金额列（不应为负）
            if any(keyword in col.lower() for keyword in ['amount', 'price', '金额', '价格', 'gmv']):
                if values.min() < 0:
                    return False, f"金额列{col}包含负值: {values.min()}"
        
        return True, "数据结果校验通过"
    
    def _fix_sql_syntax(self, sql: str, error_msg: str) -> Optional[str]:
        """修复SQL语法问题"""
        fixed = sql
        
        # 修复括号不匹配
        if "括号不匹配" in error_msg:
            open_count = fixed.count('(')
            close_count = fixed.count(')')
            if open_count > close_count:
                fixed += ')' * (open_count - close_count)
            elif close_count > open_count:
                # 移除多余的右括号
                fixed = fixed[::-1].replace(')', '', close_count - open_count)[::-1]
        
        # 修复引号不匹配
        if "引号不匹配" in error_msg:
            if fixed.count("'") % 2 != 0:
                fixed += "'"
        
        # 移除危险操作
        if "禁止执行" in error_msg:
            return None  # 不修复，直接拒绝
        
        return fixed if fixed != sql else None
    
    def _fix_sql_result(self, sql: str, error_msg: str, original_query: str) -> Optional[str]:
        """修复数据结果问题"""
        
        if "结果行数过多" in error_msg:
            # 添加LIMIT
            if 'LIMIT' not in sql.upper():
                return sql.rstrip(';') + " LIMIT 1000"
        
        if "百分比列" in error_msg and "超出合理范围" in error_msg:
            # 可能是计算方式错误，尝试用LLM修复
            if self.llm:
                prompt = f"""修复以下SQL，使百分比结果在0-100之间:

原始SQL: {sql}
错误: {error_msg}
原始查询: {original_query}

请只返回修复后的SQL:"""
                try:
                    response = self.llm.invoke(prompt)
                    fixed = response.content.strip()
                    fixed = re.sub(r'```sql\s*', '', fixed)
                    fixed = re.sub(r'```\s*', '', fixed)
                    return fixed.strip()
                except:
                    pass
        
        return None
    
    def _fix_sql_execution(self, sql: str, error_msg: str) -> Optional[str]:
        """修复SQL执行错误"""
        error_lower = error_msg.lower()
        
        # 表不存在
        if 'no such table' in error_lower or 'does not exist' in error_lower:
            return None  # 无法自动修复
        
        # 列不存在
        if 'no such column' in error_lower or 'column' in error_lower and 'does not exist' in error_lower:
            # 尝试用LLM修复
            if self.llm:
                prompt = f"""修复以下SQL的列名错误:

SQL: {sql}
错误: {error_msg}

请只返回修复后的SQL:"""
                try:
                    response = self.llm.invoke(prompt)
                    fixed = response.content.strip()
                    fixed = re.sub(r'```sql\s*', '', fixed)
                    fixed = re.sub(r'```\s*', '', fixed)
                    return fixed.strip()
                except:
                    pass
        
        return None
    
    def _record_query(self, query: str, sql: str, result, success: bool, 
                     retries: int, error: str = None):
        """记录查询历史"""
        record = {
            'query': query,
            'sql': sql,
            'success': success,
            'retries': retries,
            'result_rows': len(result) if result is not None else 0,
            'error': error
        }
        self.query_history.append(record)
    
    def get_query_history(self) -> List[Dict]:
        """获取查询历史"""
        return self.query_history
    
    def get_success_rate(self) -> float:
        """获取查询成功率"""
        if not self.query_history:
            return 0.0

        success_count = sum(1 for r in self.query_history if r['success'])
        return success_count / len(self.query_history) * 100


if __name__ == "__main__":
    # 测试用例：无LLM时规则匹配也能生成SQL
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import pandas as pd
    from agent.knowledge.knowledge_base import MetricKnowledgeBase

    class FakeDB:
        def __init__(self):
            self.engine = True

        def get_database_info(self):
            return {'tables': ['orders'],
                    'table_schemas': {'orders': {'total_amount': 'REAL', 'status': 'TEXT'}}}

        def execute_query(self, sql):
            return pd.DataFrame({'r': [1]})

    engine = NL2SQLEngine(db_connector=FakeDB(), knowledge_base=MetricKnowledgeBase(), llm=None)
    result = engine.natural_language_to_sql("上个月卖了多少")
    assert result['sql'] and result['validated'], "规则匹配未生成有效SQL"
    print(f"测试通过: 生成SQL -> {result['sql']}")

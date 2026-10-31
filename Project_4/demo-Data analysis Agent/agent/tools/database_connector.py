"""SQL数据库连接工具"""

import pandas as pd
from typing import Dict, List, Optional, Any
from sqlalchemy import create_engine, text
import logging

logger = logging.getLogger(__name__)


class DatabaseConnector:
    """SQL数据库连接器"""
    
    def __init__(self, connection_string: Optional[str] = None):
        """
        初始化数据库连接器

        Args:
            connection_string: 数据库连接字符串，例如：
                - MySQL: "mysql://user:password@host:port/dbname"（自动转 pymysql 驱动）
                - PostgreSQL: "postgresql://user:password@host:port/dbname"
                - SQLite: "sqlite:///path/to/database.db"
        """
        self.connection_string = connection_string
        self.engine = None

        if connection_string:
            self.connect(connection_string)

    @staticmethod
    def _normalize_connection_string(connection_string: str) -> str:
        """把不带驱动的 mysql:// 连接串改写为 mysql+pymysql://

        SQLAlchemy 对 mysql:// 默认使用 MySQLdb（Windows 上难以安装的 C 扩展），
        这里统一改写为纯 Python 的 pymysql 驱动。
        """
        if connection_string.startswith("mysql://"):
            return "mysql+pymysql://" + connection_string[len("mysql://"):]
        return connection_string

    def connect(self, connection_string: str) -> bool:
        """连接到数据库"""
        try:
            connection_string = self._normalize_connection_string(connection_string)
            self.engine = create_engine(connection_string)
            # 测试连接
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            self.connection_string = connection_string
            logger.info(f"✓ 数据库连接成功")
            return True
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            raise
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """执行SQL查询并返回DataFrame"""
        if not self.engine:
            raise ValueError("未连接到数据库，请先调用connect()方法")
        
        try:
            logger.info(f"执行SQL查询: {query[:100]}...")
            df = pd.read_sql_query(query, self.engine)
            logger.info(f"✓ 查询完成，返回 {len(df)} 行")
            return df
        except Exception as e:
            logger.error(f"SQL查询失败: {e}")
            raise
    
    def list_tables(self) -> List[str]:
        """列出所有表"""
        if not self.engine:
            raise ValueError("未连接到数据库")
        
        try:
            # 根据数据库类型执行不同的查询
            if 'postgresql' in self.connection_string:
                query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
            elif 'mysql' in self.connection_string:
                query = "SHOW TABLES"
            elif 'sqlite' in self.connection_string:
                query = "SELECT name FROM sqlite_master WHERE type='table'"
            else:
                raise ValueError(f"不支持的数据库类型")
            
            df = self.execute_query(query)
            tables = df.iloc[:, 0].tolist()
            logger.info(f"✓ 找到 {len(tables)} 个表")
            return tables
        except Exception as e:
            logger.error(f"列出表失败: {e}")
            raise
    
    def get_table_schema(self, table_name: str) -> Dict[str, str]:
        """获取表结构"""
        if not self.engine:
            raise ValueError("未连接到数据库")
        
        try:
            # 根据数据库类型执行不同的查询
            if 'postgresql' in self.connection_string:
                query = f"""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = '{table_name}'
                """
            elif 'mysql' in self.connection_string:
                query = f"DESCRIBE {table_name}"
            elif 'sqlite' in self.connection_string:
                query = f"PRAGMA table_info({table_name})"
            else:
                raise ValueError(f"不支持的数据库类型")
            
            df = self.execute_query(query)
            
            # 解析结果
            schema = {}
            if 'postgresql' in self.connection_string:
                for _, row in df.iterrows():
                    schema[row['column_name']] = row['data_type']
            elif 'mysql' in self.connection_string:
                for _, row in df.iterrows():
                    schema[row['Field']] = row['Type']
            elif 'sqlite' in self.connection_string:
                for _, row in df.iterrows():
                    schema[row['name']] = row['type']
            
            logger.info(f"✓ 获取表 {table_name} 的结构")
            return schema
        except Exception as e:
            logger.error(f"获取表结构失败: {e}")
            raise
    
    def load_table(self, table_name: str, limit: Optional[int] = None) -> pd.DataFrame:
        """加载表数据"""
        if not self.engine:
            raise ValueError("未连接到数据库")
        
        try:
            if limit:
                query = f"SELECT * FROM {table_name} LIMIT {limit}"
            else:
                query = f"SELECT * FROM {table_name}"
            
            df = self.execute_query(query)
            logger.info(f"✓ 加载表 {table_name}，共 {len(df)} 行")
            return df
        except Exception as e:
            logger.error(f"加载表失败: {e}")
            raise
    
    def execute_sql(self, sql: str, params: Optional[Dict] = None) -> Any:
        """执行SQL语句（INSERT, UPDATE, DELETE等）"""
        if not self.engine:
            raise ValueError("未连接到数据库")
        
        try:
            with self.engine.connect() as conn:
                if params:
                    result = conn.execute(text(sql), params)
                else:
                    result = conn.execute(text(sql))
                conn.commit()
                
                logger.info(f"✓ SQL执行成功")
                return result
        except Exception as e:
            logger.error(f"SQL执行失败: {e}")
            raise
    
    def get_database_info(self) -> Dict[str, Any]:
        """获取数据库信息"""
        if not self.engine:
            return {"status": "未连接"}
        
        try:
            tables = self.list_tables()
            table_schemas = {}
            
            for table in tables[:10]:  # 只获取前10个表的结构
                try:
                    table_schemas[table] = self.get_table_schema(table)
                except Exception as e:
                    table_schemas[table] = f"获取失败: {e}"
            
            return {
                "status": "已连接",
                "connection_string": self.connection_string,
                "tables_count": len(tables),
                "tables": tables,
                "table_schemas": table_schemas
            }
        except Exception as e:
            logger.error(f"获取数据库信息失败: {e}")
            return {"status": "错误", "error": str(e)}


if __name__ == "__main__":
    # 测试用例：验证 mysql:// 连接串归一化为 mysql+pymysql://
    s = DatabaseConnector._normalize_connection_string('mysql://user:pass@localhost:3306/db')
    assert s == 'mysql+pymysql://user:pass@localhost:3306/db', "连接串归一化失败"
    print(f"测试通过: 连接串归一化 -> {s}")

"""数据采集工具"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DataCollector:
    """电商数据采集器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.data_cache: Dict[str, pd.DataFrame] = {}
    
    def collect_from_csv(self, file_path: str, encoding: str = 'utf-8') -> pd.DataFrame:
        """从CSV文件采集数据"""
        try:
            logger.info(f"从CSV加载数据: {file_path}")
            df = pd.read_csv(file_path, encoding=encoding)
            logger.info(f"✓ CSV加载成功: {len(df)} 行, {len(df.columns)} 列")
            return df
        except Exception as e:
            logger.error(f"CSV加载失败: {e}")
            raise
    
    def collect_from_excel(self, file_path: str, sheet_name: str = 0) -> pd.DataFrame:
        """从Excel文件采集数据"""
        try:
            logger.info(f"从Excel加载数据: {file_path}")
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            logger.info(f"✓ Excel加载成功: {len(df)} 行")
            return df
        except Exception as e:
            logger.error(f"Excel加载失败: {e}")
            raise
    
    def generate_sample_data(self, n_orders: int = 100, 
                            n_customers: int = 200) -> Dict[str, pd.DataFrame]:
        """生成示例电商数据"""
        logger.info(f"生成示例数据: {n_orders} 订单, {n_customers} 客户")
        
        np.random.seed(42)
        
        # 生成客户数据
        customers = pd.DataFrame({
            'customer_id': [f'C{i:04d}' for i in range(1, n_customers + 1)],
            'age': np.random.randint(18, 65, n_customers),
            'gender': np.random.choice(['M', 'F'], n_customers),
            'city': np.random.choice(['北京', '上海', '广州', '深圳', '杭州'], n_customers),
            'signup_date': pd.date_range('2020-01-01', periods=n_customers, freq='D')
        })
        
        # 生成订单数据
        orders = pd.DataFrame({
            'order_id': [f'O{i:05d}' for i in range(1, n_orders + 1)],
            'customer_id': np.random.choice(customers['customer_id'], n_orders),
            'order_date': pd.date_range('2023-01-01', periods=n_orders, freq='h'),
            'total_amount': np.random.lognormal(mean=4.5, sigma=0.8, size=n_orders),
            'status': np.random.choice(['completed', 'cancelled', 'refunded'], 
                                      n_orders, p=[0.85, 0.10, 0.05])
        })
        
        # 生成商品数据
        n_products = 50
        products = pd.DataFrame({
            'product_id': [f'P{i:03d}' for i in range(1, n_products + 1)],
            'category': np.random.choice(['电子产品', '服装', '食品', '家居', '图书'], n_products),
            'price': np.random.lognormal(mean=4.0, sigma=0.5, size=n_products),
            'stock': np.random.randint(0, 1000, n_products)
        })
        
        sample_data = {
            'customers': customers,
            'orders': orders,
            'products': products
        }
        
        logger.info("✓ 示例数据生成完成")
        return sample_data
    
    def quality_check(self, df: pd.DataFrame) -> Dict[str, Any]:
        """数据质量检查"""
        report = {
            'shape': df.shape,
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicated_rows': int(df.duplicated().sum())
        }
        return report


if __name__ == "__main__":
    # 测试用例：生成示例数据并检查结构
    collector = DataCollector()
    data = collector.generate_sample_data(n_orders=10)
    assert 'orders' in data and len(data['orders']) == 10, "示例数据生成失败"
    print(f"测试通过: 生成数据集 {list(data.keys())}")

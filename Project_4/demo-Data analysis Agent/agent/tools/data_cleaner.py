"""数据清洗工具"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class DataCleaner:
    """电商数据清洗器"""
    
    def __init__(self):
        self.cleaning_log: List[Dict] = []
    
    def clean_pipeline(self, df: pd.DataFrame, config: Dict[str, Any] = None) -> pd.DataFrame:
        """完整清洗流程"""
        logger.info(f"开始数据清洗,原始数据: {df.shape}")
        config = config or {}
        
        # 1. 删除重复值
        if config.get('remove_duplicates', True):
            df = self.remove_duplicates(df)
        
        # 2. 处理缺失值
        if 'missing_strategy' in config:
            df = self.handle_missing_values(df, config['missing_strategy'])
        else:
            default_strategy = self._infer_missing_strategy(df)
            df = self.handle_missing_values(df, default_strategy)
        
        # 3. 处理异常值
        if 'outlier_columns' in config:
            df = self.handle_outliers(df, config['outlier_columns'])
        
        logger.info(f"清洗完成,最终数据: {df.shape}")
        return df
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """删除重复值"""
        duplicates_before = df.duplicated().sum()
        df = df.drop_duplicates().reset_index(drop=True)
        logger.info(f"删除重复值: {duplicates_before} 行")
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, 
                             strategy: Dict[str, str]) -> pd.DataFrame:
        """处理缺失值"""
        for col, method in strategy.items():
            if col not in df.columns:
                continue
            
            if method == 'mean':
                df[col] = df[col].fillna(df[col].mean())
            elif method == 'median':
                df[col] = df[col].fillna(df[col].median())
            elif method == 'mode':
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val[0])
            elif method == 'zero':
                df[col] = df[col].fillna(0)
        
        return df
    
    def handle_outliers(self, df: pd.DataFrame, columns: List[str],
                       method: str = 'iqr') -> pd.DataFrame:
        """处理异常值"""
        for col in columns:
            if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower) & (df[col] <= upper)]
        
        return df.reset_index(drop=True)
    
    def _infer_missing_strategy(self, df: pd.DataFrame) -> Dict[str, str]:
        """推断缺失值处理策略"""
        strategy = {}
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(df[col]):
                    strategy[col] = 'median'
                else:
                    strategy[col] = 'mode'
        return strategy


if __name__ == "__main__":
    # 测试用例：清洗含缺失值的数据
    import pandas as pd
    import numpy as np
    df = pd.DataFrame({'a': [1.0, 2.0, np.nan, 4.0], 'b': ['x', 'x', None, 'y']})
    cleaned = DataCleaner().clean_pipeline(df)
    assert cleaned['a'].isnull().sum() == 0, "缺失值未处理"
    print(f"测试通过: 清洗后缺失值 {cleaned.isnull().sum().sum()}")

"""数据分析工具"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class DataAnalyzer:
    """电商数据分析器"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.results: Dict[str, Any] = {}
    
    def descriptive_analysis(self) -> Dict[str, Any]:
        """描述性统计分析"""
        logger.info("执行描述性统计分析")
        
        results = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'statistics': self.df.describe().to_dict(),
            'missing_values': self.df.isnull().sum().to_dict()
        }
        
        self.results['descriptive'] = results
        return results
    
    def rfm_analysis(self, customer_col: str, date_col: str,
                    amount_col: str) -> pd.DataFrame:
        """RFM分析"""
        logger.info("执行RFM分析")

        # 确保日期列是datetime类型
        self.df[date_col] = pd.to_datetime(self.df[date_col])
        reference_date = self.df[date_col].max()
        
        # 计算RFM指标
        rfm = self.df.groupby(customer_col).agg(
            recency=(date_col, lambda x: (reference_date - x.max()).days),
            frequency=(date_col, 'count'),
            monetary=(amount_col, 'sum')
        ).reset_index()
        
        # RFM评分 (基于百分位排名,1-5分)
        rfm['R_score'] = (rfm['recency'].rank(ascending=True, pct=True) * 5).clip(1, 5).astype(int)
        rfm['R_score'] = 6 - rfm['R_score']  # recency越小越好
        rfm['F_score'] = (rfm['frequency'].rank(ascending=True, pct=True) * 5).clip(1, 5).astype(int)
        rfm['M_score'] = (rfm['monetary'].rank(ascending=True, pct=True) * 5).clip(1, 5).astype(int)
        
        rfm['RFM_score'] = rfm['R_score'] + rfm['F_score'] + rfm['M_score']
        rfm['segment'] = rfm['RFM_score'].apply(self._classify_customer)
        
        self.results['rfm'] = rfm
        logger.info(f"✓ RFM分析完成, {len(rfm)} 个客户")
        
        return rfm
    
    def _classify_customer(self, score: int) -> str:
        """根据RFM评分分类客户"""
        if score >= 13:
            return '高价值客户'
        elif score >= 10:
            return '中等价值客户'
        elif score >= 7:
            return '低价值客户'
        else:
            return '流失风险客户'
    
    def trend_analysis(self, date_col: str, value_col: str,
                      freq: str = 'ME') -> pd.DataFrame:
        """趋势分析"""
        logger.info(f"执行趋势分析, 频率={freq}")
        
        df = self.df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        
        # 按时间聚合
        trend_data = df.resample(freq, on=date_col)[value_col].sum()
        
        # 计算增长率
        growth_rate = trend_data.pct_change() * 100
        
        # 移动平均
        moving_avg = trend_data.rolling(window=3, min_periods=1).mean()
        
        results = pd.DataFrame({
            'value': trend_data,
            'growth_rate': growth_rate,
            'moving_average': moving_avg
        })
        
        self.results['trend'] = results
        logger.info(f"✓ 趋势分析完成")
        
        return results
    
    def correlation_analysis(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """相关性分析"""
        logger.info("执行相关性分析")
        
        if columns is None:
            numeric_df = self.df.select_dtypes(include=[np.number])
        else:
            numeric_df = self.df[columns]
        
        correlation_matrix = numeric_df.corr()
        self.results['correlation'] = correlation_matrix
        
        return correlation_matrix
    
    def generate_analysis_report(self) -> Dict[str, Any]:
        """生成完整分析报告"""
        report = {
            'summary': {
                'total_records': len(self.df),
                'analyses_performed': list(self.results.keys())
            },
            'results': self.results,
            'insights': self._generate_insights()
        }
        return report
    
    def _generate_insights(self) -> List[str]:
        """生成业务洞察"""
        insights = []
        
        if 'rfm' in self.results:
            rfm = self.results['rfm']
            high_value_pct = (rfm['segment'] == '高价值客户').sum() / len(rfm) * 100
            insights.append(f"高价值客户占比: {high_value_pct:.1f}%")
        
        if 'trend' in self.results:
            trend = self.results['trend']
            avg_growth = trend['growth_rate'].mean()
            if not pd.isna(avg_growth):
                insights.append(f"平均增长率: {avg_growth:.2f}%")

        return insights


if __name__ == "__main__":
    # 测试用例：RFM分析能产出客户细分
    import pandas as pd
    df = pd.DataFrame({
        'customer_id': ['C1', 'C1', 'C2'],
        'order_date': ['2024-01-01', '2024-02-01', '2024-03-01'],
        'total_amount': [100.0, 200.0, 50.0],
    })
    rfm = DataAnalyzer(df).rfm_analysis('customer_id', 'order_date', 'total_amount')
    assert 'segment' in rfm.columns and len(rfm) == 2, "RFM分析失败"
    print(f"测试通过: RFM分析产出 {len(rfm)} 个客户，细分 {rfm['segment'].tolist()}")

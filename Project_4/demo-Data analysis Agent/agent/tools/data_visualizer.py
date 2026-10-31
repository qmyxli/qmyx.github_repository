"""数据可视化工具"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from pathlib import Path
import logging
import platform

logger = logging.getLogger(__name__)


def _setup_chinese_font():
    """根据操作系统设置中文字体，避免图表中文显示为方块"""
    system = platform.system()
    if system == 'Windows':
        fonts = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    elif system == 'Darwin':  # macOS
        fonts = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC']
    else:  # Linux
        fonts = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'DejaVu Sans']
    plt.rcParams['font.sans-serif'] = fonts
    plt.rcParams['axes.unicode_minus'] = False


class DataVisualizer:
    """电商数据可视化器"""

    def __init__(self, df: pd.DataFrame, output_dir: str = './output/'):
        self.df = df
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.charts: Dict[str, str] = {}

        sns.set_style("whitegrid")
        # sns.set_style 会重置字体配置，需在之后重新应用中文字体
        _setup_chinese_font()
        plt.rcParams['figure.dpi'] = 300
    
    def plot_sales_trend(self, date_col: str, value_col: str,
                        output_path: Optional[str] = None) -> str:
        """绘制销售趋势图"""
        logger.info("绘制销售趋势图")
        
        df = self.df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        monthly_sales = df.resample('ME', on=date_col)[value_col].sum()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(monthly_sales.index, monthly_sales.values, 
               marker='o', linewidth=2, markersize=8, color='#2E86AB')
        ax.set_xlabel('时间', fontsize=12)
        ax.set_ylabel('销售额', fontsize=12)
        ax.set_title('月度销售趋势', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = output_path or str(self.output_dir / 'sales_trend.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.charts['sales_trend'] = output_path
        logger.info(f"✓ 销售趋势图已保存: {output_path}")
        return output_path
    
    def plot_customer_segments(self, rfm_df: pd.DataFrame,
                              output_path: Optional[str] = None) -> str:
        """绘制客户细分图"""
        logger.info("绘制客户细分图")
        
        segment_counts = rfm_df['segment'].value_counts()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = sns.color_palette('Set2', len(segment_counts))
        
        wedges, texts, autotexts = ax.pie(
            segment_counts.values,
            labels=segment_counts.index,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )
        
        ax.set_title('客户细分分布', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        output_path = output_path or str(self.output_dir / 'customer_segments.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.charts['customer_segments'] = output_path
        logger.info(f"✓ 客户细分图已保存: {output_path}")
        return output_path
    
    def plot_correlation_heatmap(self, columns: Optional[List[str]] = None,
                                output_path: Optional[str] = None) -> str:
        """绘制相关性热力图"""
        logger.info("绘制相关性热力图")
        
        if columns is None:
            numeric_df = self.df.select_dtypes(include=[np.number])
        else:
            numeric_df = self.df[columns]
        
        corr_matrix = numeric_df.corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', 
                   center=0, fmt='.2f', square=True, ax=ax)
        ax.set_title('特征相关性矩阵', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        output_path = output_path or str(self.output_dir / 'correlation_heatmap.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.charts['correlation_heatmap'] = output_path
        logger.info(f"✓ 相关性热力图已保存: {output_path}")
        return output_path


if __name__ == "__main__":
    # 测试用例：绘制销售趋势图并确认文件生成
    import pandas as pd
    import tempfile
    df = pd.DataFrame({
        'order_date': pd.date_range('2024-01-01', periods=60, freq='D'),
        'total_amount': range(60),
    })
    out = tempfile.mkdtemp()
    path = DataVisualizer(df, out).plot_sales_trend('order_date', 'total_amount')
    assert Path(path).exists(), "图表文件未生成"
    print(f"测试通过: 图表已生成 {path}")

"""报告生成工具"""

import json
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, analysis_results: Dict[str, Any], 
                 output_dir: str = './output/'):
        self.results = analysis_results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_markdown_report(self, output_path: Optional[str] = None) -> str:
        """生成Markdown报告"""
        logger.info("生成Markdown报告")
        
        report = []
        report.append("# 电商数据分析报告\n")
        report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append("---\n\n")
        
        # 摘要
        report.append("## 1. 执行摘要\n\n")
        if 'summary' in self.results:
            summary = self.results['summary']
            report.append(f"- **数据规模**: {summary.get('total_records', 0)} 条记录\n")
            report.append(f"- **分析维度**: {len(summary.get('analyses_performed', []))} 个\n\n")
        
        # 关键洞察
        if 'insights' in self.results:
            report.append("## 2. 关键洞察\n\n")
            for insight in self.results['insights']:
                report.append(f"- {insight}\n")
            report.append("\n")
        
        # 业务建议
        report.append("## 3. 业务建议\n\n")
        recommendations = self._generate_recommendations()
        for i, rec in enumerate(recommendations, 1):
            report.append(f"{i}. {rec}\n")
        report.append("\n")
        
        output_path = output_path or str(self.output_dir / 'report.md')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(report)
        
        logger.info(f"✓ Markdown报告已保存: {output_path}")
        return output_path
    
    def generate_html_report(self, output_path: Optional[str] = None) -> str:
        """生成HTML报告"""
        logger.info("生成HTML报告")
        
        html = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>电商数据分析报告</title>
            <style>
                body {{
                    font-family: 'Microsoft YaHei', Arial, sans-serif;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                    line-height: 1.6;
                }}
                .container {{
                    background-color: white;
                    padding: 40px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #333;
                    text-align: center;
                    border-bottom: 3px solid #2E86AB;
                    padding-bottom: 15px;
                }}
                h2 {{
                    color: #2E86AB;
                    margin-top: 30px;
                    border-left: 4px solid #2E86AB;
                    padding-left: 10px;
                }}
                .summary {{
                    background-color: #f9f9f9;
                    padding: 20px;
                    border-left: 4px solid #2E86AB;
                    margin: 20px 0;
                    border-radius: 5px;
                }}
                .insight {{
                    background-color: #e8f5e9;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 10px 0;
                    border-left: 4px solid #4CAF50;
                }}
                .recommendation {{
                    background-color: #fff3e0;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 10px 0;
                    border-left: 4px solid #FF9800;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 40px;
                    padding-top: 20px;
                    border-top: 1px solid #ddd;
                    color: #666;
                    font-size: 12px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 电商数据分析报告</h1>
                <p style="text-align: center; color: #666;">
                    生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </p>
                
                <div class="summary">
                    <h2>执行摘要</h2>
        """
        
        if 'summary' in self.results:
            summary = self.results['summary']
            html += f"""
                    <p><strong>数据规模</strong>: {summary.get('total_records', 0)} 条记录</p>
                    <p><strong>分析维度</strong>: {len(summary.get('analyses_performed', []))} 个</p>
            """
        
        html += """
                </div>
        """
        
        # 关键洞察
        if 'insights' in self.results:
            html += """
                <h2>关键洞察</h2>
            """
            for insight in self.results['insights']:
                html += f'<div class="insight">{insight}</div>'
        
        # 业务建议
        recommendations = self._generate_recommendations()
        html += """
                <h2>业务建议</h2>
        """
        for i, rec in enumerate(recommendations, 1):
            html += f'<div class="recommendation"><strong>建议{i}</strong>: {rec}</div>'
        
        html += """
                <div class="footer">
                    <p>© 2024 电商数据分析Agent | 由AI自动生成</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        output_path = output_path or str(self.output_dir / 'report.html')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"✓ HTML报告已保存: {output_path}")
        return output_path
    
    def _generate_recommendations(self) -> List[str]:
        """生成业务建议"""
        recommendations = []
        
        if 'rfm' in self.results.get('results', {}):
            rfm = self.results['results']['rfm']
            high_value_pct = (rfm['segment'] == '高价值客户').sum() / len(rfm) * 100
            
            if high_value_pct < 20:
                recommendations.append(
                    "高价值客户占比较低,建议优化客户运营策略,提升客户价值"
                )
        
        recommendations.extend([
            "优化库存管理,提高库存周转率",
            "加强数据分析能力建设,实现精细化运营",
            "关注客户体验,提升客户满意度和忠诚度"
        ])

        return recommendations


if __name__ == "__main__":
    # 测试用例：生成HTML报告并确认文件生成
    import tempfile
    results = {'summary': {'total_records': 100, 'analyses_performed': ['rfm']}, 'insights': []}
    out = tempfile.mkdtemp()
    path = ReportGenerator(results, out).generate_html_report()
    assert Path(path).exists(), "报告文件未生成"
    print(f"测试通过: 报告已生成 {path}")

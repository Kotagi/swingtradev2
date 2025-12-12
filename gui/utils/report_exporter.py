"""
Report Exporter

Export analysis results to PDF, HTML, and Excel formats.
"""

from pathlib import Path
import pandas as pd
from datetime import datetime
from typing import Dict, Optional


class ReportExporter:
    """Export analysis reports to various formats."""
    
    @staticmethod
    def export_to_html(
        trades_df: pd.DataFrame,
        metrics: Dict,
        output_path: str,
        title: str = "Backtest Analysis Report"
    ) -> bool:
        """Export backtest results to HTML report."""
        try:
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background-color: #1e1e1e;
            color: #b0b0b0;
            padding: 20px;
        }}
        h1 {{
            color: #00d4aa;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #444;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #2d2d2d;
            color: #00d4aa;
        }}
        tr:nth-child(even) {{
            background-color: #252525;
        }}
        .metric {{
            display: inline-block;
            margin: 10px;
            padding: 10px;
            background-color: #2d2d2d;
            border-radius: 5px;
        }}
        .metric-label {{
            color: #b0b0b0;
            font-size: 12px;
        }}
        .metric-value {{
            color: #00d4aa;
            font-size: 18px;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>Performance Metrics</h2>
    <div>
"""
            
            # Add metrics
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    if 'rate' in key.lower() or 'return' in key.lower():
                        value_str = f"{value:.2%}"
                    elif 'pnl' in key.lower() or 'drawdown' in key.lower() or 'capital' in key.lower():
                        value_str = f"${value:,.2f}"
                    else:
                        value_str = f"{value:.2f}"
                else:
                    value_str = str(value)
                
                html_content += f"""
        <div class="metric">
            <div class="metric-label">{key.replace('_', ' ').title()}</div>
            <div class="metric-value">{value_str}</div>
        </div>
"""
            
            html_content += """
    </div>
    
    <h2>Trade Log</h2>
"""
            
            # Add trades table
            if not trades_df.empty:
                html_content += trades_df.to_html(classes='trades-table', index=False, escape=False)
            else:
                html_content += "<p>No trades to display.</p>"
            
            html_content += """
</body>
</html>
"""
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return True
        except Exception as e:
            print(f"Error exporting to HTML: {e}")
            return False
    
    @staticmethod
    def export_to_excel(
        trades_df: pd.DataFrame,
        metrics: Dict,
        output_path: str
    ) -> bool:
        """Export backtest results to Excel file."""
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Write trades
                trades_df.to_excel(writer, sheet_name='Trades', index=False)
                
                # Write metrics
                metrics_df = pd.DataFrame([
                    {'Metric': k, 'Value': v} for k, v in metrics.items()
                ])
                metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
            
            return True
        except Exception as e:
            print(f"Error exporting to Excel: {e}")
            return False
    
    @staticmethod
    def export_to_pdf(
        trades_df: pd.DataFrame,
        metrics: Dict,
        output_path: str,
        title: str = "Backtest Analysis Report"
    ) -> bool:
        """Export backtest results to PDF (requires reportlab)."""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib import colors
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
            
            doc = SimpleDocTemplate(output_path, pagesize=letter)
            story = []
            
            styles = getSampleStyleSheet()
            story.append(Paragraph(title, styles['Title']))
            story.append(Spacer(1, 12))
            story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Metrics table
            story.append(Paragraph("Performance Metrics", styles['Heading2']))
            metrics_data = [['Metric', 'Value']]
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    if 'rate' in key.lower() or 'return' in key.lower():
                        value_str = f"{value:.2%}"
                    elif 'pnl' in key.lower() or 'drawdown' in key.lower():
                        value_str = f"${value:,.2f}"
                    else:
                        value_str = f"{value:.2f}"
                else:
                    value_str = str(value)
                metrics_data.append([key.replace('_', ' ').title(), value_str])
            
            metrics_table = Table(metrics_data)
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(metrics_table)
            story.append(Spacer(1, 20))
            
            # Trades table (limited rows for PDF)
            if not trades_df.empty:
                story.append(Paragraph("Trade Log (Sample)", styles['Heading2']))
                sample_df = trades_df.head(50)  # Limit to 50 rows for PDF
                trades_data = [list(sample_df.columns)]
                for _, row in sample_df.iterrows():
                    trades_data.append([str(val) for val in row.values])
                
                trades_table = Table(trades_data)
                trades_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('FONTSIZE', (0, 1), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(trades_table)
            
            doc.build(story)
            return True
        except ImportError:
            print("reportlab not available. Install with: pip install reportlab")
            return False
        except Exception as e:
            print(f"Error exporting to PDF: {e}")
            return False


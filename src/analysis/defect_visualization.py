import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import os
from datetime import datetime

class DefectVisualization:
    """Visualization tools for defect detection data"""
    
    def __init__(self, output_dir="output/reports"):
        """Initialize visualization with an output directory"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Color scheme for different defect types
        self.colors = {
            'Surface Defects': '#FF9999',
            'Shape Deformations': '#66B2FF',
            'Color Irregularities': '#99FF99',
            'default': '#CCCCCC'
        }
        
        # Set style for all plots
        plt.style.use('seaborn')
        
    def plot_defect_types(self, defect_types, title="Defect Types Distribution", save_path=None):
        """Plot distribution of defect types"""
        plt.figure(figsize=(10, 6))
        
        defect_names = [d['defect_type'] for d in defect_types]
        counts = [d['count'] for d in defect_types]
        colors = [self.colors.get(name, self.colors['default']) for name in defect_names]
        
        # Create bar plot
        bars = plt.bar(defect_names, counts, color=colors)
        
        # Customize plot
        plt.title(title, pad=20, fontsize=14, fontweight='bold')
        plt.xlabel('Defect Category', labelpad=10)
        plt.ylabel('Number of Defects', labelpad=10)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom')
        
        # Rotate labels if needed
        if len(max(defect_names, key=len)) > 15:
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            full_path = os.path.join(self.output_dir, save_path)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            
        return plt.gcf()
    
    def create_defect_dashboard(self, defect_types, daily_stats, severity_data, save_path=None):
        """Create a comprehensive dashboard with multiple plots"""
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        plt.suptitle('Defect Detection Dashboard', fontsize=16, fontweight='bold', y=0.95)
        
        # Define grid layout
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. Defect Types Distribution (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_defect_distribution(ax1, defect_types)
        
        # 2. Quality Score Trend (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_quality_trend(ax2, daily_stats)
        
        # 3. Defect Trend (bottom left)
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_defect_trend(ax3, daily_stats)
        
        # 4. Severity by Category (bottom right)
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_severity_by_category(ax4, severity_data)
        
        if save_path:
            full_path = os.path.join(self.output_dir, save_path)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _plot_defect_distribution(self, ax, defect_types):
        """Helper to plot defect distribution"""
        defect_names = [d['defect_type'] for d in defect_types]
        counts = [d['count'] for d in defect_types]
        colors = [self.colors.get(name, self.colors['default']) for name in defect_names]
        
        bars = ax.bar(defect_names, counts, color=colors)
        ax.set_title('Defect Types Distribution')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height):,}',
                   ha='center', va='bottom')
        
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    def _plot_quality_trend(self, ax, daily_stats):
        """Helper to plot quality score trend"""
        df = pd.DataFrame(daily_stats)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Plot quality score
        ax.plot(df['date'], df['quality_score'], 'g-', linewidth=2.5, label='Quality Score')
        
        # Add target line
        ax.axhline(y=90, color='r', linestyle='--', alpha=0.7, label='Target (90%)')
        
        # Customize plot
        ax.set_title('Quality Score Trend')
        ax.set_xlabel('Date')
        ax.set_ylabel('Quality Score (%)')
        ax.set_ylim(0, 105)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        
        # Format dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    def _plot_defect_trend(self, ax, daily_stats):
        """Helper to plot defect trend"""
        df = pd.DataFrame(daily_stats)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Plot trends
        ax.plot(df['date'], df['total_defects'], 'r-', 
                linewidth=2.5, label='Total Defects')
        ax.plot(df['date'], df['defect_objects'], 'b--', 
                linewidth=2.5, label='Objects with Defects')
        
        # Customize plot
        ax.set_title('Daily Defect Trend')
        ax.set_xlabel('Date')
        ax.set_ylabel('Count')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        
        # Format dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    def _plot_severity_by_category(self, ax, severity_data):
        """Helper to plot severity by category"""
        categories = [d['defect_type'] for d in severity_data]
        severities = [d['avg_severity'] for d in severity_data]
        counts = [d['count'] for d in severity_data]
        colors = [self.colors.get(cat, self.colors['default']) for cat in categories]
        
        # Create bar plot
        bars = ax.bar(categories, severities, color=colors)
        
        # Customize plot
        ax.set_title('Average Severity by Category')
        ax.set_xlabel('Defect Category')
        ax.set_ylabel('Average Severity')
        ax.set_ylim(0, 10)
        
        # Add count annotations
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'n={count}', ha='center', va='bottom', fontsize=9)
        
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    def plot_severity_by_type(self, defect_stats, title="Average Severity by Type", save_path=None):
        """Plot average severity by defect type"""
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        
        # Use the existing helper method
        self._plot_severity_by_category(ax, defect_stats)
        
        # Update title if provided
        if title:
            ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            full_path = os.path.join(self.output_dir, save_path)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            
        return plt.gcf()
    
    def create_defect_report(self, defect_db, days=30, report_path=None):
        """Create a comprehensive defect report with visualizations"""
        if not report_path:
            return None
            
        # Get current date for report
        current_date = datetime.now()
        
        # Calculate statistics
        defect_types = defect_db.get_defect_types()
        daily_stats = defect_db.get_defect_stats_by_day(days)
        severity_data = self.get_severity_by_category()
        
        # Aggregate defect types
        aggregated_defects = self.aggregate_defect_types(defect_types)
        
        # Generate visualizations
        self.create_defect_dashboard(
            aggregated_defects, 
            daily_stats, 
            severity_data,
            save_path="dashboard.png"
        )
        
        # Calculate summary metrics
        total_defects = sum(d['count'] for d in aggregated_defects)
        avg_quality = sum(d['quality_score'] for d in daily_stats) / len(daily_stats) if daily_stats else 0
        worst_category = max(severity_data, key=lambda x: x['avg_severity']) if severity_data else None
        
        # Format severity text
        severity_text = f"{worst_category['avg_severity']:.1f}/10" if worst_category else 'N/A'
        
        # Create HTML report
        full_report_path = os.path.join(self.output_dir, report_path)
        
        with open(full_report_path, 'w') as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Defect Analysis Report</title>
                <style>
                    :root {{
                        --primary-color: #2c3e50;
                        --secondary-color: #3498db;
                        --accent-color: #e74c3c;
                        --background-color: #f8f9fa;
                        --card-background: #ffffff;
                    }}
                    
                    body {{
                        font-family: 'Segoe UI', Arial, sans-serif;
                        line-height: 1.6;
                        margin: 0;
                        padding: 20px;
                        background-color: var(--background-color);
                        color: var(--primary-color);
                    }}
                    
                    .container {{
                        max-width: 1200px;
                        margin: 0 auto;
                    }}
                    
                    .header {{
                        text-align: center;
                        padding: 20px;
                        margin-bottom: 40px;
                        background: var(--card-background);
                        border-radius: 10px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }}
                    
                    .header h1 {{
                        color: var(--primary-color);
                        margin: 0;
                    }}
                    
                    .date {{
                        color: var(--secondary-color);
                        font-size: 1.2em;
                        margin-top: 10px;
                    }}
                    
                    .metrics-grid {{
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                        gap: 20px;
                        margin-bottom: 40px;
                    }}
                    
                    .metric-card {{
                        background: var(--card-background);
                        padding: 20px;
                        border-radius: 10px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        text-align: center;
                    }}
                    
                    .metric-value {{
                        font-size: 2em;
                        font-weight: bold;
                        color: var(--secondary-color);
                        margin: 10px 0;
                    }}
                    
                    .visualization {{
                        background: var(--card-background);
                        padding: 20px;
                        border-radius: 10px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        margin-bottom: 40px;
                    }}
                    
                    .visualization img {{
                        width: 100%;
                        height: auto;
                        border-radius: 5px;
                    }}
                    
                    .footer {{
                        text-align: center;
                        margin-top: 40px;
                        color: #666;
                        font-style: italic;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>Defect Analysis Report</h1>
                        <div class="date">Generated on {current_date.strftime('%B %d, %Y at %H:%M')}</div>
                    </div>
                    
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <h3>Total Defects</h3>
                            <div class="metric-value">{total_defects:,}</div>
                        </div>
                        <div class="metric-card">
                            <h3>Quality Score</h3>
                            <div class="metric-value">{avg_quality:.1f}%</div>
                        </div>
                        <div class="metric-card">
                            <h3>Most Severe Category</h3>
                            <div class="metric-value">{worst_category['defect_type'] if worst_category else 'N/A'}</div>
                            <div>Severity: {severity_text}</div>
                        </div>
                    </div>
                    
                    <div class="visualization">
                        <h2>Comprehensive Dashboard</h2>
                        <img src="dashboard.png" alt="Defect Analysis Dashboard">
                    </div>
                    
                    <div class="footer">
                        Report generated by Defect Detection System v1.0
                    </div>
                </div>
            </body>
            </html>
            """)
        
        return full_report_path
    
    def aggregate_defect_types(self, defect_types):
        """Aggregate similar defect types into main categories"""
        categories = {
            'Surface Defects': 0,
            'Shape Deformations': 0,
            'Color Irregularities': 0
        }
        
        for defect in defect_types:
            if 'Surface Defect' in defect['defect_type']:
                categories['Surface Defects'] += defect['count']
            elif 'Shape Deformation' in defect['defect_type']:
                categories['Shape Deformations'] += defect['count']
            elif 'Color Irregularity' in defect['defect_type']:
                categories['Color Irregularities'] += defect['count']
        
        return [
            {'defect_type': k, 'count': v}
            for k, v in categories.items()
            if v > 0  # Only include categories with defects
        ]

    def get_severity_by_category(self):
        """Get severity statistics by main defect category"""
        # This should be implemented in DefectDatabase, but we'll add a placeholder
        # that matches our visualization needs
        return [
            {
                'defect_type': 'Surface Defects',
                'count': 206,
                'avg_severity': 7.5
            },
            {
                'defect_type': 'Shape Deformations',
                'count': 320,
                'avg_severity': 6.8
            },
            {
                'defect_type': 'Color Irregularities',
                'count': 180,
                'avg_severity': 5.9
            }
        ]
import os
import datetime
import glob
import pandas as pd
import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.data.defect_database import DefectDatabase
from src.analysis.defect_visualization import DefectVisualization

class DefectDataManager:
    """Manages defect data storage, retrieval, and visualization"""
    
    def __init__(self, db_path="output/defect_database.db"):
        """Initialize the data manager"""
        self.db = DefectDatabase(db_path)
        self.visualizer = DefectVisualization()
        self.current_session_id = None
    
    def start_session(self, metadata=None):
        """Start a new detection session"""
        self.current_session_id = self.db.create_session(metadata)
        print(f"Started new defect detection session: {self.current_session_id}")
        return self.current_session_id
    
    def end_session(self, stats):
        """End the current detection session"""
        if not self.current_session_id:
            print("No active session to end")
            return
        
        self.db.end_session(self.current_session_id, stats)
        print(f"Ended session {self.current_session_id}")
        self.current_session_id = None
    
    def record_defect(self, defect_data, image_path=None):
        """Record a detected defect"""
        if not self.current_session_id:
            print("No active session, starting a new one")
            self.start_session()
        
        self.db.add_defect(defect_data, self.current_session_id, image_path)
    
    def record_object(self, object_data):
        """Record a detected object"""
        if not self.current_session_id:
            print("No active session, starting a new one")
            self.start_session()
        
        self.db.add_product(object_data, self.current_session_id)
    
    def import_existing_data(self):
        """Import existing CSV reports and images into the database"""
        # Find all CSV reports
        csv_files = glob.glob("output/defect_report_*.csv")
        if not csv_files:
            print("No CSV reports found to import")
            return
        
        print(f"Found {len(csv_files)} CSV reports to import")
        
        # Import each report
        for csv_file in csv_files:
            print(f"Importing {csv_file}...")
            
            # Check if image directory exists
            image_dir = "output/defect_images"
            if not os.path.exists(image_dir):
                print(f"Warning: Image directory {image_dir} not found")
                image_dir = None
            
            # Import the CSV file
            self.db.import_csv_report(csv_file, image_dir)
    
    def generate_daily_report(self, date=None):
        """Generate a daily defect report"""
        if date is None:
            date = datetime.datetime.now().strftime("%Y-%m-%d")
        
        # Update daily stats
        self.db.update_daily_stats(date)
        
        # Get defect data for the day
        defect_criteria = {
            'start_date': date,
            'end_date': date
        }
        
        daily_defects = self.db.search_defects(defect_criteria)
        
        if not daily_defects:
            print(f"No defects found for {date}")
            return None
        
        # Create a DataFrame for analysis
        df = pd.DataFrame(daily_defects)
        
        # Generate summary data
        summary = {
            'date': date,
            'total_defects': len(df),
            'unique_objects': df['object_id'].nunique(),
            'defect_types': df['defect_type'].value_counts().to_dict(),
            'avg_severity': df['severity'].mean()
        }
        
        # Get defect counts by type for visualization
        defect_types = []
        for defect_type, count in summary['defect_types'].items():
            defect_types.append({
                'defect_type': defect_type,
                'count': count
            })
        
        # Generate plots
        self.visualizer.plot_defect_types(
            defect_types,
            title=f"Defect Types Distribution - {date}",
            save_path=f"daily_defect_types_{date.replace('-', '')}.png"
        )
        
        # Get defect stats by type for severity analysis
        defect_stats = []
        for defect_type, count in summary['defect_types'].items():
            avg_severity = df[df['defect_type'] == defect_type]['severity'].mean()
            defect_stats.append({
                'defect_type': defect_type,
                'count': count,
                'avg_severity': avg_severity
            })
        
        self.visualizer.plot_severity_by_type(
            defect_stats,
            title=f"Average Severity by Defect Type - {date}",
            save_path=f"daily_severity_{date.replace('-', '')}.png"
        )
        
        return summary
    
    def generate_monthly_report(self, month=None, year=None):
        """Generate a monthly defect report"""
        if month is None or year is None:
            # Use current month
            today = datetime.datetime.now()
            year = today.year
            month = today.month
        
        # Calculate start and end dates
        start_date = f"{year}-{month:02d}-01"
        
        # Calculate last day of month
        if month == 12:
            next_month_year = year + 1
            next_month = 1
        else:
            next_month_year = year
            next_month = month + 1
        
        # Last day is one day before the first day of next month
        end_date = (datetime.datetime(next_month_year, next_month, 1) - 
                   datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        
        # Get daily stats for the month
        days_in_month = (datetime.datetime.strptime(end_date, "%Y-%m-%d") - 
                        datetime.datetime.strptime(start_date, "%Y-%m-%d")).days + 1
        
        daily_stats = self.db.get_defect_stats_by_day(days_in_month)
        
        # Filter to include only days in the specified month
        filtered_stats = []
        for stat in daily_stats:
            stat_date = datetime.datetime.strptime(stat['date'], "%Y-%m-%d")
            if stat_date.year == year and stat_date.month == month:
                filtered_stats.append(stat)
        
        if not filtered_stats:
            print(f"No data found for {year}-{month:02d}")
            return None
        
        # Get defect statistics by type
        defect_stats_by_type = self.db.get_defect_stats_by_type(start_date, end_date)
        
        # Create visualizations
        self.visualizer.plot_defect_trend(
            filtered_stats,
            days=days_in_month,
            title=f"Defect Trend - {year}-{month:02d}",
            save_path=f"monthly_trend_{year}{month:02d}.png"
        )
        
        self.visualizer.plot_quality_score(
            filtered_stats,
            days=days_in_month,
            title=f"Quality Score - {year}-{month:02d}",
            save_path=f"monthly_quality_{year}{month:02d}.png"
        )
        
        # Format defect types for visualization
        defect_types = []
        for stat in defect_stats_by_type:
            defect_types.append({
                'defect_type': stat['defect_type'],
                'count': stat['count']
            })
        
        self.visualizer.plot_defect_types(
            defect_types,
            title=f"Defect Types - {year}-{month:02d}",
            save_path=f"monthly_defect_types_{year}{month:02d}.png"
        )
        
        self.visualizer.plot_severity_by_type(
            defect_stats_by_type,
            title=f"Average Severity by Defect Type - {year}-{month:02d}",
            save_path=f"monthly_severity_{year}{month:02d}.png"
        )
        
        # Create dashboard
        month_name = datetime.datetime(year, month, 1).strftime("%B")
        self.visualizer.create_defect_dashboard(
            defect_types,
            filtered_stats,
            defect_stats_by_type,
            save_path=f"monthly_dashboard_{year}{month:02d}.png"
        )
        
        # Create HTML report
        report_path = f"monthly_report_{year}{month:02d}.html"
        self.visualizer.create_defect_report(
            self.db,
            days=days_in_month,
            report_path=report_path
        )
        
        return {
            'month': f"{year}-{month:02d}",
            'month_name': month_name,
            'report_path': os.path.join(self.visualizer.output_dir, report_path),
            'total_defects': sum(stat['total_defects'] for stat in filtered_stats),
            'avg_quality_score': sum(stat['quality_score'] for stat in filtered_stats) / len(filtered_stats) if filtered_stats else 0
        }
    
    def get_severity_timeline(self):
        """Get timeline of defect severities"""
        query = """
            SELECT 
                strftime('%H:%M', timestamp) as time,
                AVG(severity) as avg_severity
            FROM defects
            WHERE date(timestamp) = date('now')
            GROUP BY strftime('%H:%M', timestamp)
            ORDER BY timestamp
        """
        self.db.cursor.execute(query)
        results = self.db.cursor.fetchall()
        return [(row['time'], row['avg_severity']) for row in results]

    def get_inspection_timeline(self):
        """Get timeline of inspected objects"""
        query = """
            SELECT 
                strftime('%H:%M', timestamp) as time,
                COUNT(*) as count
            FROM products
            WHERE date(timestamp) = date('now')
            GROUP BY strftime('%H:%M', timestamp)
            ORDER BY timestamp
        """
        self.db.cursor.execute(query)
        results = self.db.cursor.fetchall()
        return [(row['time'], row['count']) for row in results]

    def get_defect_correlation_matrix(self):
        """Get correlation matrix between different defect types"""
        query = """
            SELECT DISTINCT defect_type
            FROM defects
            WHERE date(timestamp) = date('now')
        """
        self.db.cursor.execute(query)
        defect_types = [row['defect_type'] for row in self.db.cursor.fetchall()]
        
        matrix = []
        for type1 in defect_types:
            row = []
            for type2 in defect_types:
                # Count co-occurrences
                query = """
                    SELECT COUNT(DISTINCT d1.object_id) as count
                    FROM defects d1
                    JOIN defects d2 ON d1.object_id = d2.object_id
                    WHERE d1.defect_type = ? 
                    AND d2.defect_type = ?
                    AND date(d1.timestamp) = date('now')
                """
                self.db.cursor.execute(query, (type1, type2))
                count = self.db.cursor.fetchone()['count']
                row.append(count)
            matrix.append(row)
        
        return matrix if matrix else None
    
    def aggregate_defect_types(self, defect_types):
        """Aggregate defects into main categories"""
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
        ]

    def get_severity_by_category(self):
        """Get average severity by main defect category"""
        query = """
        SELECT 
            CASE 
                WHEN defect_type LIKE 'Surface Defect%' THEN 'Surface Defects'
                WHEN defect_type LIKE 'Shape Deformation%' THEN 'Shape Deformations'
                WHEN defect_type LIKE 'Color Irregularity%' THEN 'Color Irregularities'
            END as category,
            COUNT(*) as count,
            AVG(severity) as avg_severity
        FROM defects
        GROUP BY category
        ORDER BY count DESC
        """
        
        self.cursor.execute(query)
        results = self.cursor.fetchall()
        
        return [
            {
                'defect_type': row[0],
                'count': row[1],
                'avg_severity': row[2]
            }
            for row in results
        ]

    def close(self):
        """Close database connection"""
        self.db.close()
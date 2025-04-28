import argparse
import os
import sys
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import webbrowser

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.data.defect_data_manager import DefectDataManager

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Defect Analysis and Reporting Tool")
    
    # Main actions
    parser.add_argument('--import-data', action='store_true', help="Import existing CSV reports into database")
    parser.add_argument('--daily-report', action='store_true', help="Generate daily defect report")
    parser.add_argument('--monthly-report', action='store_true', help="Generate monthly defect report")
    parser.add_argument('--search', action='store_true', help="Search for specific defects")
    parser.add_argument('--dashboard', action='store_true', help="Generate and display dashboard")
    
    # Date parameters
    parser.add_argument('--date', type=str, help="Date for daily report (YYYY-MM-DD)")
    parser.add_argument('--month', type=str, help="Month for monthly report (YYYY-MM)")
    parser.add_argument('--days', type=int, default=30, help="Number of days to include in trend analysis")
    
    # Search parameters
    parser.add_argument('--defect-type', type=str, help="Filter by defect type")
    parser.add_argument('--object-id', type=int, help="Filter by object ID")
    parser.add_argument('--min-severity', type=float, help="Minimum severity")
    parser.add_argument('--max-severity', type=float, help="Maximum severity")
    parser.add_argument('--start-date', type=str, help="Start date for search range (YYYY-MM-DD)")
    parser.add_argument('--end-date', type=str, help="End date for search range (YYYY-MM-DD)")
    parser.add_argument('--limit', type=int, default=100, help="Maximum results to return")
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default="output/reports", help="Output directory for reports")
    parser.add_argument('--open', action='store_true', help="Open reports after generation")
    parser.add_argument('--db-path', type=str, default="output/defect_database.db", help="Path to database file")
    
    return parser.parse_args()

def generate_daily_report(data_manager, date=None, open_report=False):
    """Generate a daily defect report"""
    # Parse date if provided
    if date:
        try:
            date_obj = datetime.datetime.strptime(date, "%Y-%m-%d")
            date = date_obj.strftime("%Y-%m-%d")
        except ValueError:
            print(f"Error: Invalid date format. Please use YYYY-MM-DD. Got: {date}")
            return None
    
    # Generate report
    summary = data_manager.generate_daily_report(date)
    
    if summary:
        print("\nDaily Report Summary:")
        print(f"Date: {summary['date']}")
        print(f"Total defects: {summary['total_defects']}")
        print(f"Unique objects with defects: {summary['unique_objects']}")
        print(f"Average severity: {summary['avg_severity']:.2f}")
        print("\nDefect types:")
        for defect_type, count in summary['defect_types'].items():
            print(f"  {defect_type}: {count}")
        
        # Open reports if requested
        if open_report:
            image_path = os.path.join(data_manager.visualizer.output_dir, 
                                     f"daily_defect_types_{summary['date'].replace('-', '')}.png")
            if os.path.exists(image_path):
                webbrowser.open(f"file://{os.path.abspath(image_path)}")
        
        return summary
    else:
        print("No data available for the specified date")
        return None

def generate_monthly_report(data_manager, month=None, open_report=False):
    """Generate a monthly defect report"""
    year = None
    month_num = None
    
    # Parse month if provided
    if month:
        try:
            date_obj = datetime.datetime.strptime(month, "%Y-%m")
            year = date_obj.year
            month_num = date_obj.month
        except ValueError:
            print(f"Error: Invalid month format. Please use YYYY-MM. Got: {month}")
            return None
    
    # Generate report
    report_info = data_manager.generate_monthly_report(month_num, year)
    
    if report_info:
        print("\nMonthly Report Summary:")
        print(f"Month: {report_info['month_name']} {report_info['month'].split('-')[0]}")
        print(f"Total defects: {report_info['total_defects']}")
        print(f"Average quality score: {report_info['avg_quality_score']:.2f}%")
        print(f"Report saved to: {report_info['report_path']}")
        
        # Open report if requested
        if open_report and os.path.exists(report_info['report_path']):
            webbrowser.open(f"file://{os.path.abspath(report_info['report_path'])}")
        
        return report_info
    else:
        print("No data available for the specified month")
        return None

def search_defects(data_manager, search_criteria, open_report=False):
    """Search for defects based on criteria"""
    # Get database from data manager
    db = data_manager.db
    
    # Search defects
    results = db.search_defects(search_criteria)
    
    if results:
        print(f"\nFound {len(results)} defects matching criteria:")
        
        # Convert to DataFrame for easy display
        df = pd.DataFrame(results)
        
        # Format columns
        display_cols = ['id', 'object_id', 'defect_type', 'severity', 'timestamp']
        
        # Print results
        pd.set_option('display.max_rows', 20)
        print(df[display_cols].head(20))
        
        if len(df) > 20:
            print(f"... (showing 20 of {len(df)} results)")
        
        # Get summary statistics
        print("\nSummary statistics:")
        print(f"Total defects: {len(df)}")
        print(f"Unique objects: {df['object_id'].nunique()}")
        
        # Group by defect type
        type_summary = df.groupby('defect_type').agg(
            count=('id', 'count'),
            avg_severity=('severity', 'mean')
        ).reset_index()
        
        print("\nDefect types:")
        for _, row in type_summary.iterrows():
            print(f"  {row['defect_type']}: {row['count']} defects, avg severity: {row['avg_severity']:.2f}")
        
        # Create visualizations
        defect_types = []
        for _, row in type_summary.iterrows():
            defect_types.append({
                'defect_type': row['defect_type'],
                'count': row['count']
            })
        
        defect_stats = []
        for _, row in type_summary.iterrows():
            defect_stats.append({
                'defect_type': row['defect_type'],
                'count': row['count'],
                'avg_severity': row['avg_severity']
            })
        
        # Generate search criteria description for title
        title_parts = []
        if 'defect_type' in search_criteria:
            title_parts.append(f"Type: {search_criteria['defect_type']}")
        if 'object_id' in search_criteria:
            title_parts.append(f"Object: {search_criteria['object_id']}")
        if 'start_date' in search_criteria and 'end_date' in search_criteria:
            title_parts.append(f"Period: {search_criteria['start_date']} to {search_criteria['end_date']}")
        elif 'start_date' in search_criteria:
            title_parts.append(f"Since: {search_criteria['start_date']}")
        elif 'end_date' in search_criteria:
            title_parts.append(f"Until: {search_criteria['end_date']}")
        
        search_desc = " | ".join(title_parts) if title_parts else "All Defects"
        
        # Create visualizations
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        data_manager.visualizer.plot_defect_types(
            defect_types,
            title=f"Defect Types - {search_desc}",
            save_path=f"search_defect_types_{timestamp}.png"
        )
        
        data_manager.visualizer.plot_severity_by_type(
            defect_stats,
            title=f"Severity by Type - {search_desc}",
            save_path=f"search_severity_{timestamp}.png"
        )
        
        # Open report if requested
        if open_report:
            image_path = os.path.join(data_manager.visualizer.output_dir, 
                                     f"search_defect_types_{timestamp}.png")
            if os.path.exists(image_path):
                webbrowser.open(f"file://{os.path.abspath(image_path)}")
        
        return df
    else:
        print("No defects found matching the search criteria")
        return None

def generate_dashboard(data_manager, days=30, open_report=False):
    """Generate a dashboard of defect analytics"""
    # Get defect data
    defect_types = data_manager.db.get_defect_types()
    daily_stats = data_manager.db.get_defect_stats_by_day(days)
    severity_by_category = data_manager.get_severity_by_category()
    
    if not defect_types or not daily_stats:
        print("Not enough data available for dashboard generation")
        return None
    
    # Aggregate the defects into main categories
    aggregated_defects = data_manager.aggregate_defect_types(defect_types)
    
    # Create dashboard
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dashboard_path = f"dashboard_{timestamp}.png"
    
    data_manager.visualizer.create_defect_dashboard(
        aggregated_defects,
        daily_stats,
        severity_by_category,
        save_path=dashboard_path
    )
    
    # Generate HTML report
    report_path = f"dashboard_report_{timestamp}.html"
    full_report_path = data_manager.visualizer.create_defect_report(
        data_manager.db,
        days=days,
        report_path=report_path
    )
    
    if full_report_path and os.path.exists(full_report_path):
        print(f"\nDashboard Summary:")
        print(f"- Total defects grouped into 3 main categories")
        print(f"- Quality score trend over {days} days")
        print(f"- Severity analysis by defect category")
        print(f"\nOutputs saved to:")
        print(f"Dashboard: {os.path.join(data_manager.visualizer.output_dir, dashboard_path)}")
        print(f"HTML Report: {full_report_path}")
        
        # Open report if requested
        if open_report:
            webbrowser.open(f"file://{os.path.abspath(full_report_path)}")
        
        return {
            'dashboard_path': os.path.join(data_manager.visualizer.output_dir, dashboard_path),
            'report_path': full_report_path
        }
    
    return None

def main():
    """Main function"""
    args = parse_arguments()
    
    # Create data manager
    data_manager = DefectDataManager(args.db_path)
    
    # Set output directory
    data_manager.visualizer.output_dir = args.output_dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Handle each action
        if args.import_data:
            print("Importing existing defect data...")
            data_manager.import_existing_data()
        
        if args.daily_report:
            print("\nGenerating daily report...")
            generate_daily_report(data_manager, args.date, args.open)
        
        if args.monthly_report:
            print("\nGenerating monthly report...")
            generate_monthly_report(data_manager, args.month, args.open)
        
        if args.search:
            print("\nSearching for defects...")
            search_criteria = {}
            
            # Build search criteria from arguments
            if args.defect_type:
                search_criteria['defect_type'] = args.defect_type
            
            if args.object_id:
                search_criteria['object_id'] = args.object_id
            
            if args.min_severity:
                search_criteria['min_severity'] = args.min_severity
            
            if args.max_severity:
                search_criteria['max_severity'] = args.max_severity
            
            if args.start_date:
                search_criteria['start_date'] = args.start_date
            
            if args.end_date:
                search_criteria['end_date'] = args.end_date
            
            search_criteria['limit'] = args.limit
            
            search_defects(data_manager, search_criteria, args.open)
        
        if args.dashboard:
            print("\nGenerating defect dashboard...")
            generate_dashboard(data_manager, args.days, args.open)
        
        # If no action specified, show help
        if not (args.import_data or args.daily_report or args.monthly_report or 
                args.search or args.dashboard):
            print("No action specified. Use --help to see available options.")
    
    finally:
        # Close database connection
        data_manager.close()

if __name__ == "__main__":
    main()
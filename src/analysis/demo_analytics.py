#!/usr/bin/env python3
"""
Demo script to showcase the defect analytics capabilities.
This will import existing data, generate visualizations, and create reports.
"""

import os
import time
import webbrowser
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.data.defect_data_manager import DefectDataManager

def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def main():
    """Main demo function"""
    # Create output directory
    os.makedirs("output/demo", exist_ok=True)
    
    # Initialize data manager with custom output directory
    data_manager = DefectDataManager("output/defect_database.db")
    data_manager.visualizer.output_dir = "output/demo"
    
    # Step 1: Import existing data
    print_header("Step 1: Importing Existing Defect Data")
    print("Looking for defect reports in the output directory...")
    
    # Find all CSV reports
    csv_files = []
    for file in os.listdir("output"):
        if file.startswith("defect_report_") and file.endswith(".csv"):
            csv_files.append(os.path.join("output", file))
    
    if not csv_files:
        print("No defect reports found. Please run the detection system first.")
        return
    
    print(f"Found {len(csv_files)} defect reports to import.")
    
    # Import each file
    for csv_file in csv_files:
        print(f"Importing {csv_file}...")
        session_id = data_manager.db.import_csv_report(
            csv_file, 
            image_dir="output/defect_images"
        )
        print(f"Successfully imported as session {session_id}")
    
    print("\nSuccessfully imported all defect data.")
    
    # Step 2: Generate defect type visualization
    print_header("Step 2: Analyzing Defect Types")
    defect_types = data_manager.db.get_defect_types()
    
    print("Defect type distribution:")
    for defect in defect_types:
        print(f"  {defect['defect_type']}: {defect['count']} occurrences")
    
    print("\nGenerating visualization...")
    defect_chart = data_manager.visualizer.plot_defect_types(
        defect_types,
        title="Defect Types Distribution",
        save_path="defect_types_chart.png"
    )
    
    chart_path = os.path.join(data_manager.visualizer.output_dir, "defect_types_chart.png")
    print(f"Chart saved to {chart_path}")
    
    # Step 3: Generate daily stats and trend analysis
    print_header("Step 3: Analyzing Defect Trends")
    # Update daily stats first
    print("Updating daily statistics...")
    
    # Get all dates in the database
    data_manager.db.cursor.execute(
        "SELECT DISTINCT date(timestamp) as date FROM defects ORDER BY date"
    )
    dates = [row[0] for row in data_manager.db.cursor.fetchall()]
    
    for date in dates:
        data_manager.db.update_daily_stats(date)
    
    print(f"Updated statistics for {len(dates)} days")
    
    # Get daily stats
    daily_stats = data_manager.db.get_defect_stats_by_day(30)
    
    print("\nGenerating trend visualization...")
    trend_chart = data_manager.visualizer.plot_defect_trend(
        daily_stats,
        title="Defect Trend (Last 30 Days)",
        save_path="defect_trend_chart.png"
    )
    
    quality_chart = data_manager.visualizer.plot_quality_score(
        daily_stats,
        title="Quality Score Trend (Last 30 Days)",
        save_path="quality_score_chart.png"
    )
    
    # Step 4: Generate severity analysis
    print_header("Step 4: Analyzing Defect Severity")
    defect_stats = data_manager.db.get_defect_stats_by_type()
    
    print("Average severity by defect type:")
    for stat in defect_stats:
        print(f"  {stat['defect_type']}: {stat['avg_severity']:.2f} (out of 10)")
    
    print("\nGenerating severity visualization...")
    severity_chart = data_manager.visualizer.plot_severity_by_type(
        defect_stats,
        title="Average Severity by Defect Type",
        save_path="severity_chart.png"
    )
    
    # Step 5: Generate comprehensive dashboard and report
    print_header("Step 5: Creating Comprehensive Dashboard and Report")
    print("Generating dashboard...")
    
    dashboard = data_manager.visualizer.create_defect_dashboard(
        defect_types,
        daily_stats,
        defect_stats,
        save_path="dashboard.png"
    )
    
    print("Generating comprehensive HTML report...")
    report_path = data_manager.visualizer.create_defect_report(
        data_manager.db,
        days=30,
        report_path="defect_report.html"
    )
    
    print(f"Report saved to {report_path}")
    
    # Step 6: Open the report in web browser
    print_header("Step 6: Displaying Results")
    print("Opening report in web browser...")
    time.sleep(1)  # Small delay to ensure files are written
    
    if os.path.exists(report_path):
        webbrowser.open(f"file://{os.path.abspath(report_path)}")
        print("Report opened in browser")
    else:
        print(f"Error: Report file not found at {report_path}")
    
    # Clean up
    data_manager.close()
    
    print_header("Demo Completed")
    print("All visualizations and reports have been saved to:")
    print(f"  {os.path.abspath(data_manager.visualizer.output_dir)}")
    print("\nYou can analyze defect data further using the defect_analyzer.py tool.")
    print("See DATA_ANALYSIS_GUIDE.md for more information.")

if __name__ == "__main__":
    main() 
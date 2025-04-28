# Manufacturing Defect Analysis Guide

This guide explains how to use the defect data analysis and visualization tools that have been integrated with the defect detection system.

## Overview

The defect detection system now includes:

1. **Database integration**: All detected defects are stored in a SQLite database for long-term storage and analysis.
2. **Data visualization**: Generate charts, graphs, and reports from the defect data.
3. **Reporting capability**: Create daily and monthly reports with defect statistics.
4. **Standalone analysis tool**: Analyze defect data independently from the detection process.

## Quick Start

### Running Defect Detection with Data Storage

```bash
python main.py --video input.mp4 --generate-report
```

This will:
- Process the video file detecting defects
- Store all defects in the database
- Generate reports after processing finishes

### Importing Existing Data

If you have existing defect CSV reports, import them into the database:

```bash
python defect_analyzer.py --import-data
```

### Generating Reports

Generate a monthly report of defect data:

```bash
python defect_analyzer.py --monthly-report --open
```

The `--open` flag will automatically open the report in your browser after generation.

## Detailed Usage

### Main Detection Program (main.py)

The main program now supports these additional arguments:

- `--import-data`: Import existing CSV reports into the database
- `--generate-report`: Generate reports after detection
- `--no-detection`: Skip detection and only generate reports

Example:

```bash
python main.py --video input.mp4 --generate-report --save-video
```

### Defect Analyzer (defect_analyzer.py)

The standalone analyzer tool provides these features:

1. **Data Import**:
   ```bash
   python defect_analyzer.py --import-data
   ```

2. **Daily Reports**:
   ```bash
   python defect_analyzer.py --daily-report --date 2025-04-14
   ```

3. **Monthly Reports**:
   ```bash
   python defect_analyzer.py --monthly-report --month 2025-04
   ```

4. **Search Defects**:
   ```bash
   python defect_analyzer.py --search --defect-type "Scratch" --min-severity 2.5
   ```

5. **Generate Dashboard**:
   ```bash
   python defect_analyzer.py --dashboard --days 60 --open
   ```

### Search Parameters

The search feature supports these parameters:

- `--defect-type`: Filter by defect type (e.g., "Scratch", "Dent")
- `--object-id`: Filter by object ID
- `--min-severity` / `--max-severity`: Filter by severity range
- `--start-date` / `--end-date`: Filter by date range
- `--limit`: Maximum number of results to return

Example:

```bash
python defect_analyzer.py --search --defect-type "Scratch" --min-severity 2.5 --start-date 2025-04-01 --end-date 2025-04-14
```

## Database Structure

The defect database contains these main tables:

1. **defects**: Individual defect records
2. **products**: Detected objects/products
3. **sessions**: Detection sessions
4. **daily_stats**: Aggregated daily statistics

The database file is located at `output/defect_database.db`.

## Report Types

### Daily Report

Shows defects detected on a specific day, including:
- Total defect count
- Defect types distribution
- Severity analysis

### Monthly Report

Comprehensive monthly analysis with:
- Defect trends over time
- Quality score analysis
- Defect type distribution
- Severity analysis by defect type

### Dashboard

The dashboard provides a comprehensive view with:
- Defect types distribution
- Quality score trend
- Defect trend over time
- Severity analysis

## Output Files

All reports and visualizations are saved in the `output/reports` directory by default.

## Advanced Usage

### Custom Output Directory

```bash
python defect_analyzer.py --dashboard --output-dir "my_reports"
```

### Database Path

```bash
python defect_analyzer.py --monthly-report --db-path "path/to/database.db"
```

## Troubleshooting

- If no data appears in reports, make sure you've imported existing CSV data with `--import-data`
- For visualization issues, check that matplotlib and seaborn are properly installed
- Database errors may indicate permissions issues with the database file

## Integration with Other Systems

The database can be accessed by other tools using standard SQLite connections. The schema is documented in the code comments in `defect_database.py`. 
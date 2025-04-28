# Integrated Bottle Cap Inspection System

This is an integrated solution for bottle cap inspection that combines real-time defect detection, data storage, and comprehensive reporting capabilities.

## Features

- **Real-time Defect Detection**:
  - Detect bottle caps on a conveyor belt
  - Analyze shape, color, and position
  - Identify defects and anomalies
  - Support for both traditional CV and YOLOv8 detection

- **Data Management**:
  - Store all detections in a SQLite database
  - Track objects across frames
  - Save images of defective caps
  - Maintain detailed statistics

- **Comprehensive Reporting**:
  - Generate daily defect reports
  - Create monthly quality summaries
  - Visualize trends and patterns
  - Produce interactive dashboards

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Quick Start

```bash
# Run with default settings (background subtraction method)
python integrated_cap_inspection.py

# Run with YOLOv8 detection
python integrated_cap_inspection.py --use-yolo
```

### Advanced Usage

#### Generate Test Video First

```bash
python integrated_cap_inspection.py --generate-video --video-duration 60
```

#### Run with Reporting

```bash
python integrated_cap_inspection.py --use-yolo --generate-report --open-report
```

#### Full Pipeline with Dashboard

```bash
python integrated_cap_inspection.py --generate-video --use-yolo --record-video --generate-report --dashboard --open-report
```

## Detection Methods

The system supports two detection methods:

### 1. Background Subtraction (Default)

Background subtraction is a traditional computer vision technique that works well for static camera setups and controlled environments. This method:

- Uses MOG2 background subtractor
- Applies morphological operations for noise reduction
- Filters contours by size and aspect ratio
- Works without requiring a pre-trained model

### 2. YOLOv8 Detection

[YOLOv8](https://github.com/ultralytics/ultralytics) is a state-of-the-art object detection model that offers improved accuracy and robustness. Using YOLO:

- Detects objects with greater precision
- Works in varying lighting conditions
- Can recognize specific cap types and defects
- Supports custom trained models for specific applications

To use YOLOv8, add the `--use-yolo` flag:

```bash
python integrated_cap_inspection.py --use-yolo
```

## Training Custom Models

You can train a custom YOLOv8 model for your specific bottle caps:

```bash
python integrated_cap_inspection.py --use-yolo --train-yolo --data-yaml path/to/data.yaml
```

The data.yaml file should follow the YOLOv8 format and point to your training and validation images.

## Reporting

### Daily Reports

Daily reports include:
- Total defects detected
- Types of defects and their frequencies
- Average severity
- Visualizations of defect distribution

```bash
python integrated_cap_inspection.py --generate-report
```

### Monthly Reports

Monthly reports provide a comprehensive overview:
- Defect trends over time
- Quality score analysis
- Comparison across defect types
- Severity analysis

```bash
python integrated_cap_inspection.py --monthly-report --month 2023-07
```

### Dashboard

The dashboard provides a full view of your quality metrics:
- Interactive charts
- Summary statistics
- Defect galleries
- Exportable reports

```bash
python integrated_cap_inspection.py --dashboard
```

## Command Line Arguments

### Video Options
- `--video`: Path to video file or camera index (default: input.mp4)
- `--generate-video`: Generate test video before running inspection
- `--video-duration`: Duration of test video in seconds (default: 30)

### Output Options
- `--output-dir`: Output directory for all results (default: output)
- `--db-path`: Path to database file (default: output/defect_database.db)
- `--record-video`: Record inspection session to video

### Reporting Options
- `--generate-report`: Generate report after inspection
- `--open-report`: Open report in browser
- `--dashboard`: Generate dashboard after inspection

### YOLO Options
- `--use-yolo`: Use YOLOv8 for object detection
- `--yolo-model`: Path to custom YOLO model
- `--yolo-conf`: YOLO confidence threshold (default: 0.25)
- `--train-yolo`: Train a custom YOLO model before running
- `--data-yaml`: Path to data.yaml file for YOLO training

## Project Structure

- `integrated_cap_inspection.py`: Main integrated application
- `yolo_detector.py`: YOLOv8 integration for object detection
- `defect_data_manager.py`: Database and data management
- `defect_database.py`: SQLite database operations
- `defect_visualization.py`: Visualization and reporting
- `defect_analyzer.py`: Analysis of defect data
- `generate_test_video.py`: Generate test videos with simulated defects

## Troubleshooting

### YOLOv8 Issues
- Ensure ultralytics is properly installed: `pip install ultralytics`
- For CUDA errors, verify PyTorch is installed with CUDA support
- Try lowering batch size if training runs out of memory

### Database Issues
- If database errors occur, try `--db-path` to specify a new database location
- Check permissions for the output directory

### Video Source Problems
- For webcam issues, try specifying device number: `--video 0`
- For file issues, verify file exists and codec is supported 
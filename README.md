# Bottle Cap Inspection System

A computer vision-based system for bottle cap quality inspection. The system identifies bottle caps on a conveyor belt and analyzes them for defects in shape, color, and position.

## Features

- **Automated Inspection**: Automatically detects and tracks bottle caps moving in a straight line on a conveyor belt
- **Real-time Analysis**: Performs real-time shape, color, and position analysis
- **User Interface**: Displays inspection results in a clean side panel showing:
  - Cap ID (identification number)
  - Shape status (OK/DAMAGED)
  - Color status (OK/DAMAGED)
  - Position status (OK/DAMAGED)
- **Test Video Generation**: Includes a script to generate test videos with bottle caps and various defects

## Requirements

- Python 3.6+
- OpenCV
- NumPy
- Pandas (for analysis features)

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install opencv-python numpy pandas
   ```

## Usage

### Quick Start

Run the complete system (generate test video + run inspection UI):

```
python run_cap_inspection.py
```

### Options

The system can be run with various options:

```
python run_cap_inspection.py --skip-video-gen --video-output existing_video.mp4
```

Options:
- `--skip-video-gen`: Skip test video generation (use existing video)
- `--video-duration`: Set the duration of the test video in seconds (default: 30)
- `--video-output`: Specify output path for test video (default: input.mp4)
- `--output-dir`: Specify output directory for inspection results (default: output)

### Generate Test Video Only

To generate a test video without running the inspection UI:

```
python generate_test_video.py --output input.mp4 --duration 30
```

### Run Inspection UI Only

To run the inspection UI on an existing video file:

```
python cap_inspection_ui.py --video input.mp4 --output output
```

## How It Works

1. **Test Video Generation**: Creates a video with bottle caps moving on a conveyor belt
   - Caps move in a straight line
   - Some caps have defects: color variations, scratches, or misalignment

2. **Object Detection**: Uses background subtraction and contour analysis to detect objects

3. **Object Tracking**: Tracks objects across frames to maintain consistent Cap IDs

4. **Defect Analysis**:
   - **Shape Analysis**: Checks for circularity
   - **Color Analysis**: Checks for proper gold/yellow outer ring and white inner
   - **Position Analysis**: Checks if cap is properly centered

5. **UI Display**: Shows a real-time display with:
   - Left panel: Video feed with tracked objects
   - Right panel: Detailed inspection results for the cap in the central ROI

## Controls

- Press 'q' to quit the application
- Press 's' to save a screenshot of the current inspection view

## Directory Structure

- `generate_test_video.py`: Script to generate test videos with bottle caps
- `cap_inspection_ui.py`: Main inspection UI for bottle cap analysis
- `run_cap_inspection.py`: Convenience script to run both video generation and inspection
- `output/`: Default output directory for results and screenshots 
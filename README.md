# Manufacturing Vision System with Bottle Cap Inspection

A comprehensive real-time computer vision system for manufacturing environments, combining bottle cap quality inspection with advanced object detection, tracking, and segmentation. The system identifies bottle caps on a conveyor belt, analyzes them for defects in shape, color, and position, and provides detailed object analytics using YOLOv8 and SAM 2.

## Features

- **Bottle Cap Inspection**:
  - **Automated Inspection**: Detects and tracks bottle caps moving in a straight line on a conveyor belt
  - **Real-time Analysis**: Performs shape, color, and position analysis in real-time
  - **User Interface**: Displays inspection results in a clean side panel showing:
    - Cap ID (identification number)
    - Shape status (OK/DAMAGED)
    - Color status (OK/DAMAGED)
    - Position status (OK/DAMAGED)
  - **Test Video Generation**: Includes a script to generate test videos with bottle caps and various defects
- **Manufacturing Vision System**:
  - **Object Detection & Tracking**: Identifies and tracks objects using YOLOv8
  - **Precise Segmentation**: Generates detailed object outlines using SAM 2
  - **Real-time Analytics**: Monitors FPS, processing time, and object statistics
  - **Interactive UI**: Modern web-based dashboard with customizable display options
  - **Screenshot & Fullscreen**: Captures and views important moments
  - **Multi-source Input**: Supports sample videos, webcam, and custom video uploads
  - **Detailed Object Information**: Tracks IDs, positions, sizes, and confidence scores

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended for real-time performance)
- OpenCV
- NumPy
- Pandas
- Torch
- Ultralytics
- Flask
- Segment-anything

## Installation

1. Clone this repository:git clone https://github.com/SiddhantSwan12/manufacturing-vision-system.git
cd manufacturing-vision-system

2. Install dependencies:pip install -r requirements.txt

3. Verify models and sample videos:
Ensure `yolov8n.pt` and `sam2_checkpoint.pth` are in the `models/` directory. Download a sample video if needed:
python download_sample_video.py

## Usage

### Quick Start

Run the complete system (generate test video + run inspection UI + manufacturing vision dashboard):
### Options

The system can be run with various options:
Options:
- `--no-cap-inspection`: Run only the manufacturing vision dashboard, skipping bottle cap inspection
- `--skip-video-gen`: Skip test video generation (use existing video)
- `--video-duration`: Set test video duration in seconds (default: 30)
- `--video-output`: Specify output path for test video (default: input.mp4)
- `--output-dir`: Specify output directory for inspection results (default: output)

### Generate Test Video Only

To generate a test video without running the inspection UI: 
python scripts/generate_test_video.py --output input.mp4 --duration 30

### Run Inspection UI Only

To run the bottle cap inspection UI on an existing video file:
python scripts/cap_inspection_ui.py --video input.mp4 --output output

### Run Manufacturing Vision Dashboard Only

To run the web-based dashboard for object detection, tracking, and segmentation:
python run.py --no-cap-inspection

## How It Works

1. **Bottle Cap Inspection**:
   - **Test Video Generation**: Creates a video with bottle caps moving on a conveyor belt
     - Caps move in a straight line
     - Some caps have defects: color variations, scratches, or misalignment
   - **Object Detection**: Uses background subtraction and contour analysis to detect caps
   - **Object Tracking**: Tracks caps across frames to maintain consistent Cap IDs
   - **Defect Analysis**:
     - **Shape Analysis**: Checks for circularity
     - **Color Analysis**: Checks for proper gold/yellow outer ring and white inner
     - **Position Analysis**: Checks if cap is properly centered
   - **UI Display**: Shows a real-time display with:
     - Left panel: Video feed with tracked objects
     - Right panel: Detailed inspection results for the cap in the central ROI

2. **Manufacturing Vision System**:
   - **Object Detection and Tracking**: Uses YOLOv8 to identify and track objects, assigning unique track IDs
   - **Segmentation**: SAM 2 generates precise segmentation masks for detected objects
   - **Real-time Processing**: Processes video frames, applying detection, segmentation, contour extraction, and visualization
   - **Web Interface**: Streams video and displays:
     - Interactive controls
     - Real-time statistics (FPS, objects detected, processing time)
     - Detailed object information (IDs, positions, sizes, confidence scores)

## Controls

- **Bottle Cap Inspection**:
  - Press 'q' to quit the application
  - Press 's' to save a screenshot of the current inspection view
- **Web Interface**:
  - Take screenshots of the current frame
  - Toggle fullscreen mode
  - Toggle bounding boxes, contours, and track IDs
  - Monitor object details in the sidebar

## Directory Structure

- `backend/`: Core logic for detection, segmentation, and video processing
  - `detection.py`: YOLOv8 detection and tracking logic
  - `segmentation.py`: SAM 2 segmentation and contour extraction
  - `utils.py`: Helper functions for visualization and data processing
  - `video_processor.py`: Integrates detection, segmentation, and output
- `frontend/`: Web interface components
  - `static/css/style.css`: Styles for the web interface
  - `static/js/app.js`: Client-side JavaScript for interactive features
  - `templates/index.html`: HTML template for the dashboard
  - `app.py`: Flask app for serving the frontend and streaming video
- `models/`: Pre-trained models
  - `yolov8n.pt`: YOLOv8 model (downloaded automatically)
  - `sam2_checkpoint.pth`: SAM 2 checkpoint
- `data/`: Sample and uploaded videos
  - `test_video.mp4`: Sample manufacturing video
  - `uploads/`: Directory for user-uploaded videos
- `scripts/`: Bottle cap inspection scripts
  - `generate_test_video.py`: Generates test videos with bottle caps
  - `cap_inspection_ui.py`: Main inspection UI for bottle cap analysis
  - `run_cap_inspection.py`: Runs both video generation and inspection
- `output/`: Default output directory for results and screenshots
- `run.py`: Main entry point for the application
- `download_sample_video.py`: Downloads a sample video
- `requirements.txt`: List of Python dependencies
- `README.md`: Project documentation

## Customization

- **Bottle Cap Inspection**:
  - Adjust detection parameters in `scripts/cap_inspection_ui.py` (e.g., circularity thresholds, color ranges)
  - Modify test video generation settings in `scripts/generate_test_video.py` (e.g., cap size, defect frequency)
- **Manufacturing Vision System**:
  - Change model paths in `backend/detection.py` (YOLOv8) or `backend/segmentation.py` (SAM 2)
  - Adjust detection confidence threshold in `backend/detection.py`
  - Modify visualization settings (colors, line thickness) in `backend/utils.py`
  - Change resize width in `frontend/app.py` for performance vs. quality

## Troubleshooting

- **CUDA Out of Memory**: Reduce input frame size or use a smaller model
- **Slow Performance**: Use a GPU or reduce frame size for CPU processing
- **Model Loading Errors**: Verify model paths and ensure models are downloaded
- **Browser Compatibility**: Use Chrome or Firefox for best results
- **Bottle Cap Detection Issues**: Adjust detection thresholds in `scripts/cap_inspection_ui.py`
- **Missing Dependencies**: Install manually if needed:pip install opencv-python numpy pandas torch ultralytics flask segment-anything

  ## Applications in Manufacturing

- **Bottle Cap Quality Control**: Detect and flag defective caps based on shape, color, and position
- **Part Tracking**: Monitor components through assembly lines
- **Quality Control**: Identify defective parts based on visual characteristics
- **Object Counting**: Count products on a conveyor
- **Process Monitoring**: Analyze material flow through production
- **Safety Monitoring**: Detect objects in restricted areas

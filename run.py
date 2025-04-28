import os
import sys
import subprocess
import webbrowser
import time
import importlib.util
import platform
from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
from backend.video_processor import VideoProcessor
from backend.utils import get_video_info
from datetime import datetime

def check_dependency(package_name):
    """
    Check if a package is installed.
    
    Args:
        package_name: Name of the package to check
        
    Returns:
        True if installed, False otherwise
    """
    return importlib.util.find_spec(package_name) is not None

def check_dependencies():
    """
    Check if all required dependencies are installed.
    """
    dependencies = {
        'cv2': 'opencv-python',
        'torch': 'torch',
        'ultralytics': 'ultralytics',
        'flask': 'flask',
        'numpy': 'numpy'
    }
    
    missing_deps = []
    for module_name, package_name in dependencies.items():
        if not check_dependency(module_name):
            missing_deps.append(package_name)
    
    if missing_deps:
        print("Missing dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nYou can install them using:")
        print(f"pip install {' '.join(missing_deps)}")
        
        # Special note for ultralytics
        if 'ultralytics' in missing_deps:
            print("\nNote: The application can run without ultralytics, but object detection will be disabled.")
            print("If you want to use object detection, please install ultralytics.")
        
        # Ask if user wants to install dependencies automatically
        if input("\nDo you want to install missing dependencies now? (y/n): ").lower() == 'y':
            try:
                print(f"Installing: {' '.join(missing_deps)}")
                subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_deps)
                print("Dependencies installed successfully.")
                return True
            except subprocess.CalledProcessError:
                print("Failed to install dependencies. Please install them manually.")
                return False
        return False
    
    print("✓ All required dependencies are installed.")
    return True

def check_models():
    """
    Check if required models are available.
    """
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Check for YOLOv8 model
    yolo_model_path = os.path.join(models_dir, 'yolov8n.pt')
    if not os.path.exists(yolo_model_path):
        print("! YOLOv8 model not found. It will be downloaded automatically when the application starts.")
    else:
        print("✓ YOLOv8 model found.")
    
    # Check for SAM 2 model
    sam_model_path = os.path.join(models_dir, 'sam2_checkpoint.pth')
    if not os.path.exists(sam_model_path):
        print("! SAM 2 model not found. Please download it manually and place it in the 'models' directory.")
        print("  For testing purposes, the application will use a dummy SAM 2 model.")
    else:
        print("✓ SAM 2 model found.")

def check_sample_video():
    """
    Check if a sample video is available.
    """
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    video_path = os.path.join(data_dir, 'test_video.mp4')
    if not os.path.exists(video_path):
        print("! Sample video not found.")
        print("  You can download one using: python download_sample_video.py")
        print("  Alternatively, you can use your webcam or provide a path to a video file.")
        
        # Ask if user wants to download a sample video
        if input("\nDo you want to download a sample video now? (y/n): ").lower() == 'y':
            try:
                download_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'download_sample_video.py')
                if os.path.exists(download_script):
                    subprocess.check_call([sys.executable, download_script])
                    print("Sample video downloaded successfully.")
                else:
                    print("Download script not found. Please download a sample video manually.")
            except subprocess.CalledProcessError:
                print("Failed to download sample video. Please download it manually.")
    else:
        print("✓ Sample video found.")

def run_application():
    """
    Run the Flask application.
    """
    flask_app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend', 'app.py')
    
    if not os.path.exists(flask_app_path):
        print(f"! Flask application not found at {flask_app_path}")
        return False
    
    print("\nStarting the Manufacturing Vision System...")
    
    # Run the Flask app in a subprocess
    if platform.system() == 'Windows':
        # Hide console window on Windows
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        flask_process = subprocess.Popen([sys.executable, flask_app_path], 
                                         startupinfo=startupinfo)
    else:
        flask_process = subprocess.Popen([sys.executable, flask_app_path])
    
    # Wait a moment for the server to start
    print("Initializing server...")
    for _ in range(5):
        time.sleep(0.5)
        print(".", end="", flush=True)
    print("\n")
    
    # Open the web browser
    print("Opening web interface...")
    webbrowser.open('http://localhost:5000')
    
    print("\n" + "="*50)
    print("Manufacturing Vision System is now running!")
    print("Access the web interface at: http://localhost:5000")
    print("Press Ctrl+C to stop the application.")
    print("="*50 + "\n")
    
    try:
        # Keep the script running until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping the application...")
        flask_process.terminate()
        flask_process.wait()
        print("Application stopped successfully.")
    
    return True

def print_banner():
    """
    Print a banner for the application.
    """
    banner = """
    ╔═══════════════════════════════════════════════════╗
    ║                                                   ║
    ║       MANUFACTURING VISION SYSTEM                 ║
    ║       Object Detection & Tracking                 ║
    ║                                                   ║
    ╚═══════════════════════════════════════════════════╝
    """
    print(banner)

app = Flask(__name__, template_folder='frontend/templates', static_folder='frontend/static')

# Initialize video processor
try:
    from ultralytics import YOLO
    from backend.segmentation import SegmentationProcessor
    
    # Load YOLO model
    yolo_model = YOLO('yolov8n.pt')
    
    # Initialize segmentation processor
    segmentation_processor = SegmentationProcessor()
    
    # Initialize video processor with dependencies
    video_processor = VideoProcessor(yolo_model=yolo_model, segmentation_processor=segmentation_processor)
except Exception as e:
    print(f"Warning: Could not initialize object detection: {str(e)}")
    print("Running in basic mode without object detection...")
    video_processor = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/change_source', methods=['POST'])
def change_source():
    source = request.json.get('source')
    file = request.files.get('file')
    
    if source == 'sample':
        video_processor.set_source(0)  # Use sample video
    elif source == 'webcam':
        video_processor.set_source(0)  # Use webcam
    elif source == 'custom' and file:
        # Save uploaded video file
        file_path = 'uploads/custom_video.mp4'
        file.save(file_path)
        video_processor.set_source(file_path)
    
    return jsonify({'status': 'success'})

def generate_frames():
    while True:
        frame = video_processor.get_processed_frame()
        if frame is None:
            continue
            
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/get_video_info')
def video_info():
    info = get_video_info(video_processor.get_current_frame())
    return jsonify(info)

@app.route('/toggle_detection')
def toggle_detection():
    video_processor.toggle_detection()
    return jsonify({'status': 'success'})

@app.route('/toggle_tracking')
def toggle_tracking():
    video_processor.toggle_tracking()
    return jsonify({'status': 'success'})

@app.route('/toggle_statistics')
def toggle_statistics():
    video_processor.toggle_statistics()
    return jsonify({'status': 'success'})

@app.route('/capture_screenshot')
def capture_screenshot():
    frame = video_processor.get_current_frame()
    if frame is not None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'screenshot_{timestamp}.jpg'
        filepath = os.path.join('frontend/static/screenshots', filename)
        cv2.imwrite(filepath, frame)
        return jsonify({
            'status': 'success',
            'filename': filename
        })
    return jsonify({'status': 'error'})

if __name__ == "__main__":
    print_banner()
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Check models
    check_models()
    
    # Check sample video
    check_sample_video()
    
    # Run the application if dependencies are OK or user chose to continue
    if deps_ok or input("\nContinue without all dependencies? (y/n): ").lower() == 'y':
        if not run_application():
            sys.exit(1)
    else:
        print("\nPlease install the required dependencies and try again.")
        sys.exit(1) 
import os
import sys
import torch
import json
from flask import Flask, Response, render_template, request, jsonify, send_file
from datetime import datetime
import cv2
import numpy as np
import io
import threading

# Add the parent directory to the path to import backend modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.video_processor import VideoProcessor
from backend.segmentation import SegmentationProcessor
from backend.utils import list_available_cameras

# Try to import YOLO from ultralytics
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    print("Warning: ultralytics package not found. Please install it using: pip install ultralytics")
    ULTRALYTICS_AVAILABLE = False

app = Flask(__name__)

# Global display settings
display_settings = {
    'show_boxes': True,
    'show_contours': True,
    'show_ids': True
}

# Global statistics
stats = {
    'fps': 0,
    'objects_detected': 0,
    'processing_time_ms': 0,
    'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'objects': []
}

# Global variables
video_processor = None
webcam_lock = threading.Lock()
paused_frame = None
is_webcam_paused = False

# Initialize models
def initialize_models():
    # Load YOLOv8 model if available
    if ULTRALYTICS_AVAILABLE:
        yolo_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                      'models', 'yolov8n.pt')
        if not os.path.exists(yolo_model_path):
            print(f"YOLOv8 model not found at {yolo_model_path}. Downloading...")
            yolo_model = YOLO('yolov8n.pt')  # This will download the model if not present
        else:
            yolo_model = YOLO(yolo_model_path)
    else:
        # Create a dummy model if ultralytics is not available
        class DummyYOLOModel:
            def track(self, frame, persist=True):
                # Return a dummy result
                class DummyBoxes:
                    def __init__(self):
                        self.xyxy = torch.zeros((0, 4))
                        self.id = None
                
                class DummyResult:
                    def __init__(self):
                        self.boxes = DummyBoxes()
                
                return [DummyResult()]
        
        yolo_model = DummyYOLOModel()
        print("Using dummy YOLO model. Please install ultralytics for actual detection.")
    
    # Load SAM 2 model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # This is a placeholder for SAM 2 model loading
    # In a real implementation, you would load the SAM 2 model here
    # For now, we'll create a dummy model for demonstration purposes
    class DummySAM2Model:
        def __init__(self, device):
            self.device = device
        
        def predict_masks(self, image, boxes, multimask_output=False):
            # Create dummy masks of the same shape as the image
            h, w = image.shape[:2]
            masks = torch.zeros((1, h, w), dtype=torch.bool, device=self.device)
            return masks
    
    sam_model = DummySAM2Model(device)
    
    # Create segmentation processor
    segmentation_processor = SegmentationProcessor()
    
    # Create video processor with display settings
    video_processor = VideoProcessor(yolo_model, segmentation_processor, display_settings=display_settings, stats=stats)
    
    return video_processor

# Initialize the video processor
video_processor = initialize_models()

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    try:
        source = request.args.get('source', '')
        
        # Handle webcam source
        if source == 'webcam':
            return webcam_feed()
            
        # Handle video file
        video_path = request.args.get('video_path', '')
        
        if not video_path:
            # Default to a sample video if none provided
            video_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                    'data', 'test_video.mp4')
        
        if not os.path.exists(video_path):
            # Create an error frame
            error_frame = create_error_frame(f"Video file not found: {video_path}")
            _, buffer = cv2.imencode('.jpg', error_frame)
            return Response(
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n',
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
        
        # Set the video source
        if video_processor:
            success = video_processor.set_source(video_path)
            if not success:
                error_frame = create_error_frame(f"Failed to open video: {video_path}")
                _, buffer = cv2.imencode('.jpg', error_frame)
                return Response(
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n',
                    mimetype='multipart/x-mixed-replace; boundary=frame'
                )
                
            return Response(
                generate_video_frames(video_processor),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
        return "Video processor not initialized", 500
    except Exception as e:
        print(f"Error in video_feed: {str(e)}")
        error_frame = create_error_frame(f"Error: {str(e)}")
        _, buffer = cv2.imencode('.jpg', error_frame)
        return Response(
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n',
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )

def generate_video_frames(processor):
    """Generate video frames with error handling."""
    try:
        while True:
            frame = processor.get_processed_frame()
            if frame is None:
                break
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    except Exception as e:
        print(f"Error in generate_video_frames: {str(e)}")
        yield b''

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
            
        file = request.files['image']
        # Read image file
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400
            
        # Process image using video processor
        processed_image = video_processor.process_frame(image)
        
        # Convert processed image back to bytes
        _, buffer = cv2.imencode('.jpg', processed_image)
        io_buf = io.BytesIO(buffer)
        
        return send_file(
            io_buf,
            mimetype='image/jpeg',
            as_attachment=False
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/pause_webcam')
def pause_webcam():
    global is_webcam_paused, paused_frame
    try:
        with webcam_lock:
            is_webcam_paused = True
            if video_processor and video_processor.cap:
                ret, frame = video_processor.cap.read()
                if ret:
                    paused_frame = video_processor.process_frame(frame)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/resume_webcam')
def resume_webcam():
    global is_webcam_paused, paused_frame
    try:
        with webcam_lock:
            is_webcam_paused = False
            paused_frame = None
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def generate_frames():
    global is_webcam_paused, paused_frame
    while True:
        with webcam_lock:
            if is_webcam_paused and paused_frame is not None:
                # Return the paused frame
                _, buffer = cv2.imencode('.jpg', paused_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                continue
                
            if video_processor and video_processor.cap:
                ret, frame = video_processor.cap.read()
                if not ret:
                    break
                    
                # Process frame
                processed_frame = video_processor.process_frame(frame)
                
                # Convert to bytes
                _, buffer = cv2.imencode('.jpg', processed_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/webcam_feed')
def webcam_feed():
    """Webcam streaming route."""
    try:
        # Initialize webcam capture through video processor
        if video_processor:
            # Try camera indices 0, 1, and 2
            success = False
            error_messages = []
            
            for camera_idx in range(3):
                try:
                    print(f"Attempting to connect to camera {camera_idx}")
                    success = video_processor.set_source(camera_idx)
                    if success:
                        print(f"Successfully connected to camera {camera_idx}")
                        break
                except Exception as e:
                    error_messages.append(f"Camera {camera_idx}: {str(e)}")
            
            if not success:
                error_message = "No webcam found. Please check:\n"
                error_message += "1. Your webcam is properly connected\n"
                error_message += "2. Windows permissions allow camera access\n"
                error_message += "3. No other application is using the camera\n"
                error_message += "\nDetailed errors:\n" + "\n".join(error_messages)
                
                error_frame = create_error_frame(error_message)
                _, buffer = cv2.imencode('.jpg', error_frame)
                return Response(
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n',
                    mimetype='multipart/x-mixed-replace; boundary=frame'
                )
            
            return Response(
                generate_video_frames(video_processor),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
        return "Video processor not initialized", 500
    except Exception as e:
        print(f"Error in webcam_feed: {str(e)}")
        error_frame = create_error_frame(f"Error: {str(e)}")
        _, buffer = cv2.imencode('.jpg', error_frame)
        return Response(
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n',
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )

def create_error_frame(message):
    """Create an error frame with the given message."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add a red background
    frame[:, :] = (0, 0, 100)  # Dark red background
    
    # Split message into multiple lines if it's too long
    words = message.split()
    lines = []
    current_line = []
    
    for word in words:
        current_line.append(word)
        if len(' '.join(current_line)) > 40:  # Max chars per line
            lines.append(' '.join(current_line[:-1]))
            current_line = [current_line[-1]]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    # Draw each line
    y_pos = 200
    for line in lines:
        cv2.putText(frame, line, (50, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_pos += 40
    
    # Add a border
    cv2.rectangle(frame, (20, 20), (620, 460), (255, 255, 255), 2)
    
    return frame

@app.route('/update_display', methods=['GET'])
def update_display():
    """
    Update display settings.
    """
    if 'show_boxes' in request.args:
        display_settings['show_boxes'] = request.args.get('show_boxes').lower() == 'true'
    if 'show_contours' in request.args:
        display_settings['show_contours'] = request.args.get('show_contours').lower() == 'true'
    if 'show_ids' in request.args:
        display_settings['show_ids'] = request.args.get('show_ids').lower() == 'true'
    
    return jsonify(display_settings)

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """
    Handle video file upload.
    """
    if 'video' not in request.files:
        return jsonify({'success': False, 'error': 'No video file provided'})
    
    video_file = request.files['video']
    
    if video_file.filename == '':
        return jsonify({'success': False, 'error': 'No video file selected'})
    
    # Create uploads directory if it doesn't exist
    uploads_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'uploads')
    os.makedirs(uploads_dir, exist_ok=True)
    
    # Save the uploaded file
    try:
        file_path = os.path.join(uploads_dir, 'uploaded_video.mp4')
        video_file.save(file_path)
        
        # Update the video source
        global video_source
        video_source = file_path
        
        return jsonify({'success': True, 'file_path': file_path})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_stats')
def get_stats():
    """Get current statistics."""
    return jsonify(stats)

@app.route('/check_cameras')
def check_cameras():
    """Check for available camera devices."""
    try:
        available_cameras = list_available_cameras()
        return jsonify({
            'success': True,
            'available_cameras': available_cameras,
            'count': len(available_cameras)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/check_video_feed')
def check_video_feed():
    """Check if the video feed is working."""
    try:
        # Check if sample video exists
        sample_video_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                        'data', 'test_video.mp4')
        
        video_exists = os.path.exists(sample_video_path)
        
        # Check if webcam is available
        webcam_available = False
        for i in range(3):  # Try camera indices 0, 1, 2
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    webcam_available = True
                    cap.release()
                    break
                cap.release()
        
        return jsonify({
            'success': True,
            'video_exists': video_exists,
            'video_path': sample_video_path if video_exists else None,
            'webcam_available': webcam_available
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/upload_reference_image', methods=['POST'])
def upload_reference_image():
    """Upload a reference image for anomaly detection."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        reference_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if reference_image is None:
            return jsonify({'error': 'Invalid image file'}), 400

        video_processor.set_reference_image(reference_image)
        return jsonify({'success': True, 'message': 'Reference image uploaded successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/toggle_anomaly_detection', methods=['POST'])
def toggle_anomaly_detection():
    """Toggle anomaly detection on/off."""
    try:
        enabled = request.json.get('enabled', True)
        video_processor.display_settings['show_anomalies'] = enabled
        return jsonify({'success': True, 'anomaly_detection_enabled': enabled})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
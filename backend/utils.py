import cv2
import numpy as np
import time
import os
from skimage.metrics import structural_similarity as ssim

def draw_bounding_boxes(frame, detections, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes on the frame.
    
    Args:
        frame: The input frame
        detections: List of tuples (x1, y1, x2, y2, track_id, confidence, class_id)
        color: Color of the bounding box (B, G, R)
        thickness: Thickness of the bounding box lines
        
    Returns:
        Frame with bounding boxes drawn
    """
    for detection in detections:
        x1, y1, x2, y2 = detection[:4]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    return frame

def draw_contours(frame, contours_list, color=(0, 0, 255), thickness=2):
    """
    Draw contours on the frame.
    
    Args:
        frame: The input frame
        contours_list: List of contours for each detection
        color: Color of the contours (B, G, R)
        thickness: Thickness of the contour lines
        
    Returns:
        Frame with contours drawn
    """
    for contours in contours_list:
        cv2.drawContours(frame, contours, -1, color, thickness)
    
    return frame

def draw_track_ids(frame, detections, centroids, color=(255, 0, 0), font_scale=0.5, thickness=1):
    """
    Draw track IDs on the frame.
    
    Args:
        frame: The input frame
        detections: List of tuples (x1, y1, x2, y2, track_id, confidence, class_id)
        centroids: List of centroids (x, y) for each detection
        color: Color of the text (B, G, R)
        font_scale: Scale of the font
        thickness: Thickness of the text
        
    Returns:
        Frame with track IDs drawn
    """
    for i, detection in enumerate(detections):
        if len(detection) >= 5 and i < len(centroids):
            track_id = detection[4]
            if track_id != -1:
                cx, cy = centroids[i]
                cv2.putText(frame, f"ID: {int(track_id)}", (cx, cy), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    
    return frame

def calculate_fps(start_time, frame_count):
    """
    Calculate frames per second.
    
    Args:
        start_time: Start time in seconds
        frame_count: Number of frames processed
        
    Returns:
        FPS value
    """
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    return round(fps, 1)

def resize_frame(frame, width=None, height=None):
    """
    Resize a frame while maintaining aspect ratio.
    
    Args:
        frame: The input frame
        width: Target width (if None, calculated from height)
        height: Target height (if None, calculated from width)
        
    Returns:
        Resized frame
    """
    if width is None and height is None:
        return frame
    
    h, w = frame.shape[:2]
    
    if width is None:
        aspect_ratio = height / float(h)
        dim = (int(w * aspect_ratio), height)
    else:
        aspect_ratio = width / float(w)
        dim = (width, int(h * aspect_ratio))
    
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

def get_object_details(detections, centroids):
    """
    Extract detailed information about detected objects.
    
    Args:
        detections: List of tuples (x1, y1, x2, y2, track_id, confidence, class_id)
        centroids: List of centroids (x, y) for each detection
        
    Returns:
        List of dictionaries containing object details
    """
    objects = []
    
    for i, detection in enumerate(detections):
        if i < len(centroids):
            x1, y1, x2, y2 = detection[:4]
            track_id = detection[4] if len(detection) > 4 else -1
            confidence = detection[5] if len(detection) > 5 else 1.0
            class_id = detection[6] if len(detection) > 6 else 0
            
            width = int(x2 - x1)
            height = int(y2 - y1)
            cx, cy = centroids[i]
            
            object_info = {
                "id": int(track_id) if track_id != -1 else i,
                "position": {"x": int(cx), "y": int(cy)},
                "size": {"width": width, "height": height},
                "confidence": float(confidence),
                "class_id": int(class_id)
            }
            
            objects.append(object_info)
    
    return objects

def add_timestamp(frame, color=(255, 255, 255), font_scale=0.5, thickness=1):
    """
    Add timestamp to the frame.
    
    Args:
        frame: The input frame
        color: Color of the text (B, G, R)
        font_scale: Scale of the font
        thickness: Thickness of the text
        
    Returns:
        Frame with timestamp
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    h, w = frame.shape[:2]
    cv2.putText(frame, timestamp, (10, h - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    return frame

def list_available_cameras(max_cameras=10):
    """
    List available camera devices.
    
    Args:
        max_cameras: Maximum number of cameras to check
        
    Returns:
        List of available camera indices
    """
    available_cameras = []
    for i in range(max_cameras):
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_ANY)
            if cap.isOpened():
                # Try to read a frame to verify camera works
                ret, frame = cap.read()
                if ret and frame is not None:
                    available_cameras.append(i)
                cap.release()
        except Exception as e:
            print(f"Error checking camera {i}: {str(e)}")
            continue
    
    return available_cameras

def initialize_camera(camera_id):
    """
    Initialize a camera with optimal settings.
    
    Args:
        camera_id: ID of the camera to initialize
        
    Returns:
        Tuple of (success, capture_object)
    """
    try:
        # Try different backends
        for backend in [cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_MSMF]:
            cap = cv2.VideoCapture(camera_id, backend)
            if cap.isOpened():
                # Set camera properties
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                
                # Verify camera works by reading a test frame
                ret, frame = cap.read()
                if ret and frame is not None:
                    return True, cap
                cap.release()
        
        return False, None
    except Exception as e:
        print(f"Error initializing camera {camera_id}: {str(e)}")
        return False, None

def get_video_info(frame):
    """
    Get information about the video frame.
    
    Args:
        frame: The current video frame
        
    Returns:
        Dictionary containing video information
    """
    if frame is None:
        return {
            "resolution": {"width": 0, "height": 0},
            "channels": 0,
            "status": "No frame available"
        }
    
    height, width = frame.shape[:2]
    channels = frame.shape[2] if len(frame.shape) > 2 else 1
    
    return {
        "resolution": {
            "width": int(width),
            "height": int(height)
        },
        "channels": int(channels),
        "status": "Active"
    }

def detect_anomalies(reference_image, target_image):
    """
    Detect anomalies by comparing a reference image with a target image.

    Args:
        reference_image: The original reference image.
        target_image: The image to compare against the reference.

    Returns:
        A tuple (anomaly_mask, similarity_score).
    """
    # Resize target image to match reference image
    target_image = cv2.resize(target_image, (reference_image.shape[1], reference_image.shape[0]))

    # Convert to grayscale
    ref_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

    # Compute structural similarity
    similarity_score, diff = ssim(ref_gray, target_gray, full=True)
    diff = (diff * 255).astype("uint8")

    # Threshold the difference to create an anomaly mask
    _, anomaly_mask = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)

    return anomaly_mask, similarity_score

def track_defect(defect_info, frame, timestamp):
    """
    Track and store defect information.
    """
    defect_record = {
        'timestamp': timestamp,
        'defect_type': defect_info['defect_type'],
        'severity': defect_info['severity'],
        'confidence': defect_info['confidence'],
        'probabilities': defect_info['probabilities']
    }
    
    # Save defect image
    if frame is not None:
        save_dir = os.path.join('data', 'defects')
        os.makedirs(save_dir, exist_ok=True)
        filename = f"defect_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(os.path.join(save_dir, filename), frame)
        defect_record['image_path'] = filename
    
    return defect_record
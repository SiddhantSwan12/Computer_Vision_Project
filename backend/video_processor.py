import cv2
import time
import numpy as np
from datetime import datetime
from .detection import detect_and_track
from .defect_detection import DefectClassifier
from .utils import (
    draw_bounding_boxes, 
    draw_contours, 
    draw_track_ids, 
    calculate_fps, 
    resize_frame,
    get_object_details,
    add_timestamp,
    initialize_camera,
    detect_anomalies,
    track_defect  # Import detect_anomalies function
)

class VideoProcessor:
    def __init__(self, yolo_model=None, segmentation_processor=None, resize_width=640, display_settings=None, stats=None):
        """
        Initialize the video processor.
        
        Args:
            yolo_model: YOLOv8 model for detection and tracking (optional)
            segmentation_processor: Segmentation processor for generating masks (optional)
            resize_width: Width to resize frames to (None for no resizing)
            display_settings: Dictionary of display settings (show_boxes, show_contours, show_ids)
            stats: Dictionary to store statistics
        """
        self.yolo_model = yolo_model
        self.segmentation_processor = segmentation_processor
        self.resize_width = resize_width
        self.frame_count = 0
        self.start_time = None
        self.display_settings = display_settings or {
            'show_boxes': True,
            'show_contours': True,
            'show_ids': True
        }
        self.stats = stats or {
            'fps': 0,
            'objects_detected': 0,
            'processing_time_ms': 0,
            'objects': []
        }
        self.cap = None  # Video capture object
        self.source = None  # Current video source
        self.reference_image = None  # Reference image for anomaly detection
        self.defect_classifier = DefectClassifier()
        self.defect_history = []

    def set_source(self, source):
        """Set the video source (file path or camera index)"""
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            
            self.source = source
            
            # If source is an integer, it's a camera index
            if isinstance(source, int):
                # Try different backends for webcam
                backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
                success = False
                
                for backend in backends:
                    try:
                        print(f"Trying camera {source} with backend {backend}")
                        self.cap = cv2.VideoCapture(source, backend)
                        if self.cap.isOpened():
                            # Configure camera settings
                            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                            self.cap.set(cv2.CAP_PROP_FPS, 30)
                            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                            
                            # Test if we can read a frame
                            ret, test_frame = self.cap.read()
                            if ret and test_frame is not None:
                                print(f"Successfully initialized camera {source} with backend {backend}")
                                success = True
                                break
                            
                        if self.cap is not None:
                            self.cap.release()
                            self.cap = None
                    except Exception as e:
                        print(f"Error with backend {backend}: {str(e)}")
                        if self.cap is not None:
                            self.cap.release()
                            self.cap = None
                        continue
                
                if not success:
                    raise Exception(f"Failed to initialize camera {source} with any backend")
            else:
                # Source is a file path
                self.cap = cv2.VideoCapture(str(source))
                if not self.cap.isOpened():
                    raise Exception(f"Failed to open video file: {source}")
            
            # Reset counters
            self.frame_count = 0
            self.start_time = time.time()
            
            return True
            
        except Exception as e:
            print(f"Error in set_source: {str(e)}")
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            return False

    def get_current_frame(self):
        """Get the current frame without processing"""
        try:
            if self.cap is None or not self.cap.isOpened():
                return None
                
            for _ in range(3):  # Try up to 3 times to read a frame
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    return frame
                time.sleep(0.1)  # Short delay between retries
                
            # If we failed to get a frame after retries, try to reset the camera
            if isinstance(self.source, int):
                print("Attempting to reset camera connection...")
                self.set_source(self.source)
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    return frame
                    
            return None
        except Exception as e:
            print(f"Error in get_current_frame: {str(e)}")
            return None

    def get_processed_frame(self):
        """Get the current frame with processing applied"""
        try:
            frame = self.get_current_frame()
            if frame is None:
                return None
                
            # Resize frame if needed
            if self.resize_width is not None:
                frame = resize_frame(frame, width=self.resize_width)
                
            return self.process_frame(frame)
        except Exception as e:
            print(f"Error in get_processed_frame: {str(e)}")
            return None

    def process_video(self, video_path):
        """
        Process a video file and yield annotated frames.
        
        Args:
            video_path: Path to the video file
            
        Yields:
            JPEG-encoded frames
        """
        cap = cv2.VideoCapture(video_path)
        self.start_time = time.time()
        self.frame_count = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            # Resize frame if needed
            if self.resize_width is not None:
                frame = resize_frame(frame, width=self.resize_width)
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Encode frame for streaming
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            self.frame_count += 1
        
        cap.release()
    
    def process_frame(self, frame):
        """
        Process a single frame.
        
        Args:
            frame: The input frame
            
        Returns:
            Annotated frame
        """
        # Start timing for processing time calculation
        frame_start_time = time.time()
        
        # Create a copy of the frame for processing
        processed_frame = frame.copy()
        
        # Initialize empty lists for detections and masks
        detections = []
        masks = []
        centroids = []
        
        # Only perform detection and tracking if YOLO model is available
        if self.yolo_model is not None:
            try:
                # Run detection and tracking
                results = self.yolo_model.track(frame, persist=True)
                if results and len(results) > 0:
                    # Extract bounding boxes and track IDs
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes.xyxy) > 0:
                        for i in range(len(boxes.xyxy)):
                            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                            track_id = boxes.id[i].item() if boxes.id is not None else -1
                            confidence = boxes.conf[i].item() if boxes.conf is not None else 1.0
                            class_id = boxes.cls[i].item() if boxes.cls is not None else 0
                            detections.append([x1, y1, x2, y2, track_id, confidence, class_id])
            except Exception as e:
                print(f"Error in object detection: {str(e)}")
        
        # Only perform segmentation if processor is available
        if self.segmentation_processor is not None and detections:
            try:
                # Get bounding boxes without track IDs
                boxes = [det[:4] for det in detections]
                
                # Generate segmentation masks
                masks = self.segmentation_processor.get_segmentation(frame, boxes)
                
                # Get contours and centroids
                contours_list, centroids = self.segmentation_processor.get_contours_and_centroids(masks, frame.shape)
                
                # Draw visualizations based on display settings
                if self.display_settings.get('show_boxes', True):
                    processed_frame = draw_bounding_boxes(processed_frame, detections)
                
                if self.display_settings.get('show_contours', True):
                    processed_frame = draw_contours(processed_frame, contours_list)
                
                if self.display_settings.get('show_ids', True):
                    processed_frame = draw_track_ids(processed_frame, detections, centroids)
            except Exception as e:
                print(f"Error in segmentation: {str(e)}")
        
        # Perform anomaly detection if reference image is available
        if self.reference_image is not None:
            try:
                anomaly_mask, similarity_score = detect_anomalies(self.reference_image, frame)
                if self.display_settings.get('show_anomalies', True):
                    # Highlight anomalies on the frame
                    frame[anomaly_mask > 0] = [0, 0, 255]  # Red color for anomalies
            except Exception as e:
                print(f"Error in anomaly detection: {str(e)}")

        # Perform defect detection on detected objects
        if detections and self.defect_classifier:
            for i, detection in enumerate(detections):
                x1, y1, x2, y2 = map(int, detection[:4])
                roi = frame[y1:y2, x1:x2]
                
                # Get defect information
                defect_info = self.defect_classifier.detect_defects(roi)
                
                if defect_info and defect_info['defect_type'] != 'no_defect':
                    # Track defect
                    defect_record = track_defect(
                        defect_info, 
                        roi,
                        datetime.now()
                    )
                    self.defect_history.append(defect_record)
                    
                    # Visualize defect
                    color = (0, 0, 255) if defect_info['severity'] == 'high' else \
                           (0, 255, 255) if defect_info['severity'] == 'medium' else \
                           (0, 255, 0)
                    
                    cv2.putText(processed_frame, 
                              f"{defect_info['defect_type']} ({defect_info['severity']})",
                              (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Calculate and update statistics
        if self.stats:
            fps = calculate_fps(self.start_time, self.frame_count)
            processing_time = (time.time() - frame_start_time) * 1000
            
            self.stats.update({
                'fps': round(fps, 1),
                'objects_detected': len(detections),
                'processing_time_ms': round(processing_time, 1),
                'objects': get_object_details(detections, centroids) if centroids else []
            })
        
        # Add timestamp to frame
        processed_frame = add_timestamp(processed_frame)
        
        # Increment frame counter
        self.frame_count += 1
        
        return processed_frame

    def set_reference_image(self, image):
        """Set the reference image for anomaly detection."""
        self.reference_image = image

    def toggle_detection(self):
        """Toggle object detection on/off"""
        if self.display_settings:
            self.display_settings['show_boxes'] = not self.display_settings['show_boxes']

    def toggle_tracking(self):
        """Toggle object tracking on/off"""
        if self.display_settings:
            self.display_settings['show_ids'] = not self.display_settings['show_ids']

    def toggle_statistics(self):
        """Toggle statistics display on/off"""
        if self.display_settings:
            self.display_settings['show_stats'] = not self.display_settings.get('show_stats', True)

    def __del__(self):
        """Clean up resources"""
        if self.cap is not None:
            self.cap.release()

    def process_webcam(self, camera_id=0):
        """
        Process webcam feed and yield annotated frames.
        
        Args:
            camera_id: ID of the camera to use
            
        Yields:
            JPEG-encoded frames
        """
        # Try to open the webcam
        cap = cv2.VideoCapture(camera_id)
        
        # Check if webcam opened successfully
        if not cap.isOpened():
            # If webcam failed to open, yield an error message frame
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, "Error: Could not access webcam", (50, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(error_frame, "Please check your camera connection", (50, 280), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Encode error frame for streaming
            ret, buffer = cv2.imencode('.jpg', error_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            return
        
        self.start_time = time.time()
        self.frame_count = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                # If frame reading failed, create an error frame
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_frame, "Error: Lost connection to webcam", (50, 240), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Encode error frame for streaming
                ret, buffer = cv2.imencode('.jpg', error_frame)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                break
            
            # Resize frame if needed
            if self.resize_width is not None:
                frame = resize_frame(frame, width=self.resize_width)
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Encode frame for streaming
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            self.frame_count += 1
        
        cap.release()
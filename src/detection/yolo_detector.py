import cv2
import numpy as np
import os
import torch
from ultralytics import YOLO
import supervision as sv
from datetime import datetime

class YOLODetector:
    """YOLO object detector for bottle cap inspection"""
    
    def __init__(self, model_path=None, conf_threshold=0.25, device=None):
        """Initialize YOLO detector
        
        Args:
            model_path: Path to pre-trained YOLO model (can be YOLOv8n.pt or custom model)
            conf_threshold: Confidence threshold for detection (0-1)
            device: Device to use (None for auto-selection, 'cpu', 'cuda:0', etc.)
        """
        self.conf_threshold = conf_threshold
        
        # Set device
        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Load model
        if model_path is None or not os.path.exists(model_path):
            print("No model specified, loading YOLOv8n...")
            self.model = YOLO('yolov8n.pt')
        else:
            print(f"Loading model from {model_path}...")
            self.model = YOLO(model_path)
        
        # Set model parameters
        self.model.conf = self.conf_threshold
        
        # Create annotator
        self.box_annotator = sv.BoxAnnotator(
            thickness=2,
            text_thickness=2,
            text_scale=1
        )
    
    def detect(self, frame):
        """Detect objects in a frame
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            detections: List of detection dictionaries (bbox, confidence, class_id)
            annotated_frame: Frame with annotations drawn
        """
        # Run YOLO detection
        results = self.model(frame, verbose=False)[0]
        
        # Convert to supervision detections
        detections = sv.Detections.from_ultralytics(results)
        
        # Filter detections if needed
        if len(detections) > 0 and self.model.names:
            # Get detection labels
            labels = [
                f"{self.model.names[int(class_id)]} {confidence:.2f}"
                for class_id, confidence in zip(detections.class_id, detections.confidence)
            ]
            
            # Annotate frame
            annotated_frame = self.box_annotator.annotate(
                scene=frame.copy(),
                detections=detections,
                labels=labels
            )
        else:
            annotated_frame = frame.copy()
        
        return detections, annotated_frame
    
    def analyze_defects(self, frame, detections):
        """Analyze detected objects for defects
        
        Args:
            frame: Original frame
            detections: Detection results from YOLO
            
        Returns:
            defects: List of defects found
        """
        defects = []
        
        if len(detections) == 0:
            return defects
        
        # Extract detection information
        bboxes = detections.xyxy  # (N, 4) - xmin, ymin, xmax, ymax
        class_ids = detections.class_id  # (N,)
        confidences = detections.confidence  # (N,)
        
        for i, bbox in enumerate(bboxes):
            # Get class ID and name
            class_id = class_ids[i]
            confidence = confidences[i]
            
            # Extract ROI
            x1, y1, x2, y2 = map(int, bbox)
            roi_img = frame[y1:y2, x1:x2]
            
            if roi_img.size == 0:
                continue
                
            # Assign a unique object ID (can be improved with tracking)
            object_id = i + 1
            
            # Timestamp for detection
            timestamp = datetime.now().isoformat()
            
            # Perform additional analysis on the ROI (can be customized)
            # This is a simple example - you can expand with more sophisticated analysis
            has_defect, defect_type, severity = self._analyze_roi(roi_img, class_id)
            
            if has_defect:
                defects.append({
                    'object_id': object_id,
                    'defect_type': defect_type,
                    'severity': severity,
                    'bbox': (x1, y1, x2-x1, y2-y1),
                    'confidence': float(confidence),
                    'class_id': int(class_id),
                    'timestamp': timestamp,
                    'roi_image': roi_img
                })
        
        return defects
    
    def _analyze_roi(self, roi_img, class_id):
        """Analyze ROI for defects (simplified method)
        
        In a real implementation, this would be more sophisticated,
        potentially using another model or computer vision techniques.
        """
        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        
        # Example: Check for color consistency
        h_std = np.std(hsv[:,:,0])
        s_std = np.std(hsv[:,:,1])
        v_std = np.std(hsv[:,:,2])
        
        # Large variation in color might indicate defects
        color_variation = (h_std + s_std + v_std) / 3
        
        # Convert to grayscale for shape analysis
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        
        # Simple edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_count = np.count_nonzero(edges)
        edge_ratio = edge_count / (roi_img.shape[0] * roi_img.shape[1])
        
        # Check for defects
        has_defect = False
        defect_type = "no_defect"
        severity = 0.0
        
        # Color defect
        if color_variation > 30:
            has_defect = True
            defect_type = "color_variation"
            severity = min(10.0, color_variation / 10)
        
        # Shape defect (if many edges relative to area)
        if edge_ratio > 0.2:
            # Only override if more severe
            if edge_ratio * 10 > severity:
                has_defect = True
                defect_type = "shape_defect"
                severity = min(10.0, edge_ratio * 10)
        
        return has_defect, defect_type, severity
    
    def train_custom_model(self, data_yaml, epochs=50, imgsz=640, batch=16, name="yolov8n_custom"):
        """Train a custom YOLO model for bottle cap detection
        
        Args:
            data_yaml: Path to data.yaml file defining dataset
            epochs: Number of epochs to train
            imgsz: Image size for training
            batch: Batch size
            name: Name for the trained model
            
        Returns:
            model_path: Path to the trained model
        """
        if not os.path.exists(data_yaml):
            print(f"Error: data.yaml not found at {data_yaml}")
            return None
        
        print(f"Training custom YOLO model with {data_yaml}...")
        
        # Create a new model instance for training
        model = YOLO('yolov8n.pt')  # Start from pre-trained model
        
        # Train the model
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            name=name
        )
        
        # Get the path to the best model
        model_path = os.path.join('runs', 'detect', name, 'weights', 'best.pt')
        
        if os.path.exists(model_path):
            print(f"Trained model saved to {model_path}")
            # Update the current model
            self.model = YOLO(model_path)
            return model_path
        else:
            print("Training completed but model file not found")
            return None


# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = YOLODetector()
    
    # Open video source (can be camera or file)
    cap = cv2.VideoCapture(0)  # 0 for webcam
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        detections, annotated_frame = detector.detect(frame)
        
        # Analyze for defects
        defects = detector.analyze_defects(frame, detections)
        
        # Display defect count
        cv2.putText(
            annotated_frame, 
            f"Defects: {len(defects)}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 0, 255), 
            2
        )
        
        # Show frame
        cv2.imshow("YOLO Bottle Cap Detection", annotated_frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
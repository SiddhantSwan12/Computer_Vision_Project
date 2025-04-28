try:
    from ultralytics import YOLO
    import numpy as np
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    import numpy as np
    print("Warning: ultralytics package not found. Please install it using: pip install ultralytics")
    ULTRALYTICS_AVAILABLE = False

import cv2
from collections import deque

# Global history of tracked objects
TRACK_HISTORY = {}
MAX_HISTORY_LENGTH = 50  # Maximum number of frames to keep history

def detect_and_track(frame, model):
    """
    Detect and track objects in a frame using YOLOv8.
    
    Args:
        frame: The input frame
        model: The YOLOv8 model
        
    Returns:
        List of tuples (x1, y1, x2, y2, track_id) for each detection
    """
    global TRACK_HISTORY

    if not ULTRALYTICS_AVAILABLE:
        # Return empty detections if ultralytics is not available
        print("Ultralytics not available. Returning empty detections.")
        return []
    
    results = model.track(frame, persist=True)
    detections = []
    
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id
        
        if track_ids is not None:
            track_ids = track_ids.cpu().numpy()
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                track_id = int(track_ids[i]) if i < len(track_ids) else -1

                # Match with history for consistent IDs
                if track_id != -1:
                    TRACK_HISTORY[track_id] = deque(maxlen=MAX_HISTORY_LENGTH)
                    TRACK_HISTORY[track_id].append((x1, y1, x2, y2))

                detections.append((float(x1), float(y1), float(x2), float(y2), track_id))

    # Remove old entries from history
    TRACK_HISTORY = {k: v for k, v in TRACK_HISTORY.items() if len(v) > 0}

    return detections
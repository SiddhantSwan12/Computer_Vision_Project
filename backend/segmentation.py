import cv2
import numpy as np

class SegmentationProcessor:
    def __init__(self):
        """
        Initialize the segmentation processor.
        """
        pass
    
    def get_segmentation(self, frame, boxes):
        """
        Generate segmentation masks for detected objects.
        
        Args:
            frame: Input frame
            boxes: List of bounding boxes (x1, y1, x2, y2)
            
        Returns:
            List of binary masks
        """
        masks = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            
            # Extract ROI
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            
            # Convert to grayscale
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Simple thresholding for basic segmentation
            _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # Create full-size mask
            full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            full_mask[y1:y2, x1:x2] = mask
            
            masks.append(full_mask)
        
        return masks
    
    def get_contours_and_centroids(self, masks, frame_shape):
        """
        Extract contours and calculate centroids from segmentation masks.
        
        Args:
            masks: List of binary masks
            frame_shape: Shape of the original frame
            
        Returns:
            Tuple of (contours_list, centroids)
        """
        contours_list = []
        centroids = []
        
        for mask in masks:
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_list.append(contours)
            
            # Calculate centroid
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    # Fallback to bounding box center if moments fail
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    cx = x + w // 2
                    cy = y + h // 2
                centroids.append((cx, cy))
            else:
                # If no contours found, use center of the frame
                h, w = frame_shape[:2]
                centroids.append((w // 2, h // 2))
        
        return contours_list, centroids 
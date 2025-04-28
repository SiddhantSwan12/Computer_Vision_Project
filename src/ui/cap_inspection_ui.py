import cv2
import numpy as np
import argparse
import os
from datetime import datetime

class CapInspectionUI:
    def __init__(self, video_source=0, output_dir="output"):
        # Create output directories
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "defect_images"), exist_ok=True)
        
        # Open video source
        if isinstance(video_source, str) and os.path.isfile(video_source):
            self.cap = cv2.VideoCapture(video_source)
        else:
            self.cap = cv2.VideoCapture(video_source)
            
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video source: {video_source}")
            
        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video opened successfully. Resolution: {self.frame_width}x{self.frame_height}")
        
        # Set up central ROI for inspection
        roi_width = int(self.frame_width * 0.3)
        roi_height = int(self.frame_height * 0.3)
        self.central_roi = {
            'x': (self.frame_width - roi_width) // 2,
            'y': (self.frame_height - roi_height) // 2,
            'width': roi_width,
            'height': roi_height
        }
        
        # Initialize object tracking
        self.next_object_id = 1
        self.tracked_objects = {}
        self.current_object_in_roi = None
        
        # Initialize background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)
        
        # Define UI parameters
        self.inspection_ui_width = 400
        self.ui_height = self.frame_height
        self.ui_width = self.frame_width + self.inspection_ui_width
        
        # Defect history
        self.defect_history = []
        self.object_history = []
        
    def detect_objects(self, frame):
        """Detect objects using background subtraction and contour analysis"""
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply additional noise removal
        fg_mask = cv2.medianBlur(fg_mask, 5)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size
        min_area = 500  # Minimum contour area to be considered an object
        max_area = self.frame_width * self.frame_height * 0.5  # Maximum area (50% of frame)
        
        filtered_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if min_area < area < max_area:
                # Additional filtering based on aspect ratio
                x, y, w, h = cv2.boundingRect(c)
                aspect_ratio = float(w) / h
                
                # Accept objects with reasonable aspect ratios
                if 0.5 < aspect_ratio < 2.0:
                    filtered_contours.append(c)
        
        return filtered_contours
    
    def is_in_central_roi(self, bbox):
        """Check if an object is in the central ROI"""
        x, y, w, h = bbox
        obj_center_x = x + w//2
        obj_center_y = y + h//2
        
        # Check if center point is in central ROI
        roi = self.central_roi
        return (roi['x'] <= obj_center_x <= roi['x'] + roi['width'] and 
                roi['y'] <= obj_center_y <= roi['y'] + roi['height'])
    
    def analyze_object_properties(self, roi_img):
        """Analyze object properties (shape, color, position)"""
        if roi_img is None or roi_img.size == 0:
            return {'shape': 'Unknown', 'color': 'Unknown', 'position': (0, 0)}
            
        # Make a copy to avoid modifying the original
        analysis_img = roi_img.copy()
        
        # Initialize default values
        shape = "Unknown"
        color = "Unknown"
        position = (0, 0)
        shape_status = "ERROR"
        color_status = "ERROR"
        position_status = "ERROR"
        
        try:
            # Convert to grayscale for shape analysis
            gray = cv2.cvtColor(analysis_img, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to separate object from background
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest contour (assumed to be the object)
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Approximate shape
                perimeter = cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, 0.04 * perimeter, True)
                
                # Get bounding circle for circularity check
                (x, y), radius = cv2.minEnclosingCircle(largest_contour)
                center = (int(x), int(y))
                
                # Get contour area and the area of the enclosing circle
                contour_area = cv2.contourArea(largest_contour)
                circle_area = np.pi * radius * radius
                
                # Check circularity
                circularity = contour_area / circle_area if circle_area > 0 else 0
                
                # For bottle caps, we expect a circle shape
                if circularity > 0.8:
                    shape = "Circle"
                    shape_status = "OK"
                else:
                    shape = "Not Circle"
                    shape_status = "DAMAGED"
                
                # Color analysis using HSV
                hsv = cv2.cvtColor(analysis_img, cv2.COLOR_BGR2HSV)
                
                # Create mask of the object
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.drawContours(mask, [largest_contour], -1, 255, -1)
                
                # Check for gold/yellow color typical for bottle caps
                # Gold/yellow hue ranges from approximately 20-40 in HSV
                lower_gold = np.array([15, 100, 100])
                upper_gold = np.array([45, 255, 255])
                
                gold_mask = cv2.inRange(hsv, lower_gold, upper_gold)
                gold_mask = cv2.bitwise_and(gold_mask, mask)
                
                # Check for white inner part
                lower_white = np.array([0, 0, 200])
                upper_white = np.array([180, 30, 255])
                
                white_mask = cv2.inRange(hsv, lower_white, upper_white)
                white_mask = cv2.bitwise_and(white_mask, mask)
                
                # Calculate percentages
                gold_percentage = (cv2.countNonZero(gold_mask) / cv2.countNonZero(mask)) if cv2.countNonZero(mask) > 0 else 0
                white_percentage = (cv2.countNonZero(white_mask) / cv2.countNonZero(mask)) if cv2.countNonZero(mask) > 0 else 0
                
                # Determine color status
                if gold_percentage > 0.3 and white_percentage > 0.2:
                    color = "Gold/White"
                    color_status = "OK"
                else:
                    color = "Irregular"
                    color_status = "DAMAGED"
                
                # Check position relative to ROI center
                roi_center_x = roi_img.shape[1] // 2
                roi_center_y = roi_img.shape[0] // 2
                
                # Calculate distance from center
                distance = np.sqrt((center[0] - roi_center_x)**2 + (center[1] - roi_center_y)**2)
                
                # Position is OK if it's close to the center
                if distance < roi_img.shape[1] * 0.2:  # Within 20% of width
                    position = "Centered"
                    position_status = "OK"
                else:
                    position = "Off-center"
                    position_status = "DAMAGED"
                
                # Return detailed analysis results
                position = center
                
        except Exception as e:
            print(f"Error analyzing object: {str(e)}")
        
        return {
            'shape': shape,
            'color': color,
            'position': position,
            'shape_status': shape_status,
            'color_status': color_status,
            'position_status': position_status
        }
    
    def track_objects(self, frame, contours):
        """Track detected objects across frames"""
        current_tracked = {}
        
        # Check existing tracked objects against new detections
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if this object corresponds to an existing tracked object
            new_object = True
            for obj_id, obj_info in self.tracked_objects.items():
                tx, ty, tw, th = obj_info['bbox']
                
                # Check for overlap
                if (x < tx + tw and x + w > tx and
                    y < ty + th and y + h > ty):
                    # This is likely the same object
                    current_tracked[obj_id] = {
                        'bbox': (x, y, w, h),
                        'contour': contour,
                        'frames_tracked': obj_info['frames_tracked'] + 1
                    }
                    new_object = False
                    break
            
            # If no matching object was found, create a new one
            if new_object:
                current_tracked[self.next_object_id] = {
                    'bbox': (x, y, w, h),
                    'contour': contour,
                    'frames_tracked': 1
                }
                self.next_object_id += 1
        
        # Update tracked objects
        self.tracked_objects = current_tracked
        return current_tracked
    
    def create_inspection_ui(self, main_frame, object_in_roi=None):
        """Create the inspection UI panel showing cap properties and inspection results"""
        # Create a white background for the UI panel
        ui_panel = np.ones((self.ui_height, self.inspection_ui_width, 3), dtype=np.uint8) * 255
        
        # Draw header
        cv2.putText(ui_panel, "Cap Inspection Results", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        
        # Draw separator line
        cv2.line(ui_panel, (20, 60), (self.inspection_ui_width - 20, 60), (200, 200, 200), 2)
        
        # If there is an object in ROI, display its properties
        if object_in_roi:
            # Extract object image
            roi_x, roi_y, roi_w, roi_h = object_in_roi['bbox']
            roi_img = main_frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w].copy()
            
            # Resize ROI image to fit in UI
            display_size = 220
            display_img = cv2.resize(roi_img, (display_size, display_size))
            
            # Display object image
            h_offset = 80
            ui_panel[h_offset:h_offset+display_size, 
                    (self.inspection_ui_width-display_size)//2:(self.inspection_ui_width+display_size)//2] = display_img
            
            # Draw object ID
            cv2.putText(ui_panel, f"Cap ID:", (40, h_offset + display_size + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
            cv2.putText(ui_panel, f"{object_in_roi['id']}", (200, h_offset + display_size + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            # Draw inspection results using the format from the reference image
            properties = self.analyze_object_properties(roi_img)
            
            # Shape status
            y_offset = h_offset + display_size + 80
            cv2.putText(ui_panel, "Shape:", (40, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
            status_color = (0, 255, 0) if properties['shape_status'] == "OK" else (0, 0, 255)
            cv2.putText(ui_panel, properties['shape_status'], (200, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            # Color status
            y_offset += 40
            cv2.putText(ui_panel, "Color:", (40, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
            status_color = (0, 255, 0) if properties['color_status'] == "OK" else (0, 0, 255)
            cv2.putText(ui_panel, properties['color_status'], (200, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            # Position status
            y_offset += 40
            cv2.putText(ui_panel, "Position:", (40, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
            status_color = (0, 255, 0) if properties['position_status'] == "OK" else (0, 0, 255)
            cv2.putText(ui_panel, properties['position_status'], (200, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            # Store analysis results with the object
            object_in_roi['properties'] = properties
            
            # Add timestamp for when this object was analyzed
            object_in_roi['timestamp'] = datetime.now()
            
            # Store object in history if not already there
            if not any(obj['id'] == object_in_roi['id'] for obj in self.object_history):
                self.object_history.append(object_in_roi)
                
        else:
            # No object in ROI
            cv2.putText(ui_panel, "No Object Detected", (50, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 100), 2)
        
        return ui_panel
    
    def process_frame(self, frame):
        """Process a video frame and create the inspection UI"""
        # Create a copy for visualization
        display_frame = frame.copy()
        
        # Detect objects
        object_contours = self.detect_objects(frame)
        
        # Track objects
        tracked_objects = self.track_objects(frame, object_contours)
        
        # Draw central ROI
        roi = self.central_roi
        cv2.rectangle(display_frame, 
                     (roi['x'], roi['y']), 
                     (roi['x'] + roi['width'], roi['y'] + roi['height']), 
                     (0, 255, 0), 2)
        
        # Add ROI label
        cv2.putText(display_frame, "Inspection Zone", 
                   (roi['x'], roi['y'] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Object in central ROI
        self.current_object_in_roi = None
        
        # Draw bounding boxes and process objects
        for obj_id, obj_info in tracked_objects.items():
            x, y, w, h = obj_info['bbox']
            
            # Draw bounding box
            cv2.rectangle(display_frame, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)
            cv2.putText(display_frame, f"ID: {obj_id}", (int(x), int(y) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Check if object is in central ROI
            if self.is_in_central_roi((x, y, w, h)):
                # Highlight object in central ROI
                cv2.rectangle(display_frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 3)
                
                # Extract ROI image
                roi_img = frame[int(y):int(y+h), int(x):int(x+w)]
                if roi_img.size == 0:  # Skip if ROI is empty
                    continue
                
                # Store object in ROI for inspection UI
                self.current_object_in_roi = {
                    'id': obj_id,
                    'bbox': (x, y, w, h),
                    'timestamp': datetime.now(),
                    'roi_image': roi_img.copy() if roi_img.size > 0 else None
                }
        
        # Create the inspection UI panel
        inspection_ui = self.create_inspection_ui(frame, self.current_object_in_roi)
        
        # Combine main frame and inspection UI
        combined_ui = np.zeros((self.ui_height, self.ui_width, 3), dtype=np.uint8)
        combined_ui[:, :self.frame_width] = display_frame
        combined_ui[:, self.frame_width:] = inspection_ui
        
        return combined_ui
    
    def run(self):
        """Run the cap inspection UI"""
        print("Starting Cap Inspection UI")
        print("Press 'q' to quit, 's' to save a screenshot")
        
        while True:
            # Read a frame
            ret, frame = self.cap.read()
            if not ret:
                print("End of video stream")
                break
            
            # Process the frame
            ui_frame = self.process_frame(frame)
            
            # Display the UI
            cv2.imshow("Cap Inspection System", ui_frame)
            
            # Process key events
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = os.path.join(self.output_dir, f"inspection_{timestamp}.jpg")
                cv2.imwrite(screenshot_path, ui_frame)
                print(f"Screenshot saved to {screenshot_path}")
        
        # Release everything when done
        self.cap.release()
        cv2.destroyAllWindows()
        print("Cap Inspection UI closed")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Bottle Cap Inspection UI')
    parser.add_argument('--video', type=str, default='input.mp4', help='Path to video file or camera index')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    
    args = parser.parse_args()
    
    # Check if the video source is a file or camera index
    video_source = args.video
    if video_source.isdigit():
        video_source = int(video_source)
    
    # Create and run the inspection UI
    ui = CapInspectionUI(video_source=video_source, output_dir=args.output)
    ui.run()

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
import os
import argparse
import sys
import cv2
import numpy as np
import datetime
import time
import webbrowser


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.data.defect_data_manager import DefectDataManager
from src.detection.yolo_detector import YOLODetector
from src.analysis.defect_analyzer import generate_dashboard 

class IntegratedCapInspection:
    def __init__(self, video_source=0, output_dir="output", db_path="output/defect_database.db", 
                 use_yolo=False, yolo_model=None, yolo_conf=0.25):
        """Initialize the integrated cap inspection system with reporting capabilities."""
        # Create output directories
        self.output_dir = output_dir
        self.defect_images_dir = os.path.join(output_dir, "defect_images")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.defect_images_dir, exist_ok=True)
        
        # Initialize data manager for reporting
        self.data_manager = DefectDataManager(db_path)
        self.data_manager.visualizer.output_dir = os.path.join(output_dir, "reports")
        os.makedirs(self.data_manager.visualizer.output_dir, exist_ok=True)
        
        # Start detection session
        self.session_id = self.data_manager.start_session({
            'source': str(video_source),
            'start_time': datetime.datetime.now().isoformat()
        })
        
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
        
        # Initialize detection method
        self.use_yolo = use_yolo
        if use_yolo:
            print("Using YOLOv8 for object detection")
            self.yolo_detector = YOLODetector(
                model_path=yolo_model, 
                conf_threshold=yolo_conf
            )
        else:
            print("Using background subtraction for object detection")
            # Initialize background subtractor for motion detection
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=100, 
                varThreshold=50, 
                detectShadows=False
            )
        
        # Define UI parameters
        self.inspection_ui_width = 400
        self.ui_height = self.frame_height
        self.ui_width = self.frame_width + self.inspection_ui_width
        
        # Initialize detection statistics
        self.total_objects = 0
        self.defect_objects = 0
        self.total_defects = 0
        self.defect_counts = {}
        
        # Defect history
        self.defect_history = []
        self.object_history = []
        
        # Optionally set up video writer
        self.video_writer = None
    
    def detect_objects(self, frame):
        """Detect objects using background subtraction or YOLO"""
        if self.use_yolo:
            # Use YOLOv8 for detection
            detections, _ = self.yolo_detector.detect(frame)
            
            # Convert YOLO detections to contours for consistency
            contours = []
            if len(detections) > 0:
                # Extract bounding boxes
                bboxes = detections.xyxy  # (N, 4) - xmin, ymin, xmax, ymax
                
                for bbox in bboxes:
                    x1, y1, x2, y2 = map(int, bbox)
                    # Create a contour (rectangle) from the bbox
                    cnt = np.array([
                        [[x1, y1]],
                        [[x2, y1]],
                        [[x2, y2]],
                        [[x1, y2]]
                    ])
                    contours.append(cnt)
            
            return contours
        else:
            # Use background subtraction
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
        """Analyze object properties with specific defect types"""
        if roi_img is None or roi_img.size == 0:
            return {'defect_types': ['Invalid ROI']}

        defect_types = []
        
        # Convert to HSV and grayscale for analysis
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        
        # Shape Analysis
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(largest_contour, True)
            area = cv2.contourArea(largest_contour)
            
            # Check circularity
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            if circularity < 0.85:
                defect_types.append("Shape Deformation (Circularity: {:.2f})".format(circularity))
            
            # Check symmetry
            moments = cv2.moments(largest_contour)
            if moments['m00'] != 0:
                cx = moments['m10'] / moments['m00']
                cy = moments['m01'] / moments['m00']
                if abs(cx - roi_img.shape[1]/2) > 20 or abs(cy - roi_img.shape[0]/2) > 20:
                    defect_types.append("Position Misalignment")
        
        # Color Analysis
        h_std = np.std(hsv[:,:,0])
        s_std = np.std(hsv[:,:,1])
        v_std = np.std(hsv[:,:,2])
        color_variation = (h_std + s_std + v_std) / 3
        
        if color_variation > 30:
            defect_types.append("Color Irregularity (Variation: {:.2f})".format(color_variation))
        
        # Surface Analysis
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.count_nonzero(edges) / (roi_img.shape[0] * roi_img.shape[1])
        if edge_density > 0.1:
            defect_types.append("Surface Defect (Edge Density: {:.2f})".format(edge_density))
        
        result = {
            'defect_types': defect_types if defect_types else ['No Defects Detected'],
            'metrics': {
                'circularity': circularity if 'circularity' in locals() else 0,
                'color_variation': color_variation,
                'edge_density': edge_density
            }
        }
        
        return result
    
    def track_objects(self, frame, contours):
        """Track detected objects across frames using centroid tracking"""
        current_tracked = {}
        
        # Calculate centroids for current detections
        current_objects = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            centroid = (int(x + w/2), int(y + h/2))
            current_objects.append({
                'bbox': (x, y, w, h),
                'centroid': centroid,
                'contour': contour
            })
        
        # If no objects are currently being tracked, add all as new
        if not self.tracked_objects:
            for obj in current_objects:
                current_tracked[self.next_object_id] = {
                    'bbox': obj['bbox'],
                    'centroid': obj['centroid'],
                    'contour': obj['contour'],
                    'frames_tracked': 1,
                    'last_seen': 0  # Frame counter
                }
                self.next_object_id += 1
        else:
            # Calculate distances between current and tracked objects
            for obj in current_objects:
                min_dist = float('inf')
                matched_id = None
                
                # Find the closest tracked object
                for obj_id, tracked_obj in self.tracked_objects.items():
                    dist = np.sqrt(
                        (obj['centroid'][0] - tracked_obj['centroid'][0])**2 +
                        (obj['centroid'][1] - tracked_obj['centroid'][1])**2
                    )
                    
                    # If within reasonable distance, consider it the same object
                    if dist < 50 and dist < min_dist:  # 50 pixels threshold
                        min_dist = dist
                        matched_id = obj_id
                
                if matched_id is not None:
                    # Update existing object
                    current_tracked[matched_id] = {
                        'bbox': obj['bbox'],
                        'centroid': obj['centroid'],
                        'contour': obj['contour'],
                        'frames_tracked': self.tracked_objects[matched_id]['frames_tracked'] + 1,
                        'last_seen': 0
                    }
                else:
                    # Add new object
                    current_tracked[self.next_object_id] = {
                        'bbox': obj['bbox'],
                        'centroid': obj['centroid'],
                        'contour': obj['contour'],
                        'frames_tracked': 1,
                        'last_seen': 0
                    }
                    self.next_object_id += 1
                    self.total_objects += 1
        
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
            if roi_img.size > 0:
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
                cv2.putText(ui_panel, "Defects:", (40, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
                for defect in properties['defect_types']:
                    y_offset += 40
                    cv2.putText(ui_panel, defect, (40, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                
                # Store analysis results with the object
                object_in_roi['properties'] = properties
                
                # Add timestamp for when this object was analyzed
                object_in_roi['timestamp'] = datetime.datetime.now()
                
                # Check if object has defects
                has_defect = len(properties['defect_types']) > 0 and properties['defect_types'][0] != 'No Defects Detected'
                
                # Record defects to database
                if has_defect:
                    # Save defect image
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    defect_img_path = os.path.join(self.defect_images_dir, 
                                                 f"defect_{object_in_roi['id']}_{timestamp}.jpg")
                    cv2.imwrite(defect_img_path, roi_img)
                    
                    for defect in properties['defect_types']:
                        self.record_defect(object_in_roi['id'], defect, 7.5, defect_img_path)
                
                # Store object in history if not already there
                if not any(obj['id'] == object_in_roi['id'] for obj in self.object_history):
                    self.object_history.append(object_in_roi)
                    
                    # Record object in database
                    self.record_object(object_in_roi['id'], has_defect)
                
        else:
            # No object in ROI
            cv2.putText(ui_panel, "No Object Detected", (50, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 100), 2)
        
        return ui_panel
    
    def record_defect(self, object_id, defect_type, severity, image_path=None):
        """Record a defect in the database"""
        # Create defect data structure
        defect_data = {
            'object_id': object_id,
            'defect_type': defect_type,
            'severity': severity,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Record in data manager
        self.data_manager.record_defect(defect_data, image_path)
        
        # Update statistics
        self.total_defects += 1
        self.defect_counts[defect_type] = self.defect_counts.get(defect_type, 0) + 1
        
        # Add to defect history
        self.defect_history.append({
            'object_id': object_id,
            'defect_type': defect_type,
            'severity': severity,
            'timestamp': datetime.datetime.now()
        })
    
    def record_object(self, object_id, has_defect):
        """Record an object in the database"""
        # Create object data structure
        object_data = {
            'id': object_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'defect': {
                'has_defect': has_defect
            }
        }
        
        # Record in data manager
        self.data_manager.record_object(object_data)
        
        # Update statistics
        if has_defect:
            self.defect_objects += 1
    
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
                    'timestamp': datetime.datetime.now(),
                    'roi_image': roi_img.copy() if roi_img.size > 0 else None
                }
        
        # Create the inspection UI panel
        inspection_ui = self.create_inspection_ui(frame, self.current_object_in_roi)
        
        # Add statistics to the UI
        stats_y = self.ui_height - 100
        cv2.putText(inspection_ui, f"Total Objects: {self.total_objects}", (20, stats_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(inspection_ui, f"Defective Objects: {self.defect_objects}", (20, stats_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(inspection_ui, f"Total Defects: {self.total_defects}", (20, stats_y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Combine main frame and inspection UI
        combined_ui = np.zeros((self.ui_height, self.ui_width, 3), dtype=np.uint8)
        combined_ui[:, :self.frame_width] = display_frame
        combined_ui[:, self.frame_width:] = inspection_ui
        
        # Write frame to video if writer is active
        if self.video_writer:
            self.video_writer.write(combined_ui)
        
        return combined_ui 

    def run(self):
        """Run the cap inspection UI"""
        print("Starting Integrated Cap Inspection System")
        print("Press 'q' to quit, 's' to save a screenshot, 'r' to generate report")
        
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
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = os.path.join(self.output_dir, f"inspection_{timestamp}.jpg")
                cv2.imwrite(screenshot_path, ui_frame)
                print(f"Screenshot saved to {screenshot_path}")
            elif key == ord('r'):
                # Generate quick report
                print("Generating quick report...")
                self.generate_report()
        
        # End session and clean up
        self.end_session()
        
        # Release everything when done
        self.cap.release()
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()
        print("Cap Inspection System closed")
    
    def end_session(self):
        """End the current detection session and update statistics"""
        stats = {
            'total_objects': self.total_objects,
            'defect_objects': self.defect_objects,
            'total_defects': self.total_defects
        }
        
        self.data_manager.end_session(stats)
        print(f"Session ended. Total objects: {self.total_objects}, Defects: {self.total_defects}")
    
    def setup_video_recording(self, output_path="output/inspection_recording.mp4"):
        """Set up video recording"""
        # Define codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            output_path, fourcc, 20.0, (self.ui_width, self.ui_height))
        print(f"Video recording set up: {output_path}")
    
    def generate_report(self, open_report=True):
        """Generate comprehensive defect report with multiple visualizations"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        from datetime import datetime, timedelta
        import pandas as pd
        
        # Create reports directory if it doesn't exist
        reports_dir = os.path.join(self.output_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        # Get current date for the report
        date = datetime.now().strftime("%Y-%m-%d")
        
        # Get data from data manager
        summary = self.data_manager.generate_daily_report(date)
        
        if summary:
            # Create a figure with multiple subplots
            plt.style.use('seaborn')
            fig = plt.figure(figsize=(20, 15))
            
            # 1. Defect Types Distribution (Pie Chart)
            ax1 = plt.subplot2grid((3, 3), (0, 0))
            defect_types = list(summary['defect_types'].keys())
            defect_counts = list(summary['defect_types'].values())
            ax1.pie(defect_counts, labels=defect_types, autopct='%1.1f%%')
            ax1.set_title('Defect Types Distribution')
            
            # 2. Defect Severity Timeline (Line Plot)
            ax2 = plt.subplot2grid((3, 3), (0, 1), colspan=2)
            severity_data = self.data_manager.get_severity_timeline()
            if severity_data:
                times, severities = zip(*severity_data)
                ax2.plot(times, severities, marker='o')
                ax2.set_title('Defect Severity Timeline')
                ax2.set_xlabel('Time')
                ax2.set_ylabel('Severity')
                plt.xticks(rotation=45)
            
            # 3. Object Inspection Timeline (Bar Plot)
            ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
            inspection_data = self.data_manager.get_inspection_timeline()
            if inspection_data:
                times, counts = zip(*inspection_data)
                ax3.bar(times, counts)
                ax3.set_title('Objects Inspected Over Time')
                ax3.set_xlabel('Time')
                ax3.set_ylabel('Number of Objects')
                plt.xticks(rotation=45)
            
            # 4. Defect Occurrence Heatmap
            ax4 = plt.subplot2grid((3, 3), (1, 2))
            defect_matrix = self.data_manager.get_defect_correlation_matrix()
            if defect_matrix is not None:
                sns.heatmap(defect_matrix, annot=True, cmap='YlOrRd', ax=ax4)
                ax4.set_title('Defect Correlation Heatmap')
            
            # 5. Top Defect Categories (Horizontal Bar Chart)
            ax5 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
            sorted_defects = sorted(summary['defect_types'].items(), 
                                  key=lambda x: x[1], reverse=True)
            defect_names, defect_values = zip(*sorted_defects)
            ax5.barh(defect_names, defect_values)
            ax5.set_title('Top Defect Categories')
            ax5.set_xlabel('Number of Occurrences')
            
            # Adjust layout and save
            plt.tight_layout()
            report_path = os.path.join(reports_dir, f'defect_analysis_report_{date}.png')
            plt.savefig(report_path, dpi=300, bbox_inches='tight')
            
            # Generate summary text report
            text_report_path = os.path.join(reports_dir, f'defect_analysis_report_{date}.txt')
            with open(text_report_path, 'w') as f:
                f.write("=== Defect Analysis Report ===\n\n")
                f.write(f"Date: {date}\n")
                f.write(f"Total defects: {summary['total_defects']}\n")
                f.write(f"Unique objects with defects: {summary['unique_objects']}\n")
                f.write(f"Average severity: {summary['avg_severity']:.2f}\n\n")
                f.write("Defect types:\n")
                for defect_type, count in summary['defect_types'].items():
                    f.write(f"  {defect_type}: {count}\n")
            
            print("\nReport Summary:")
            print(f"Date: {summary['date']}")
            print(f"Total defects: {summary['total_defects']}")
            print(f"Unique objects with defects: {summary['unique_objects']}")
            print(f"Average severity: {summary['avg_severity']:.2f}")
            print("\nDefect types:")
            for defect_type, count in summary['defect_types'].items():
                print(f"  {defect_type}: {count}")
            
            # Open reports if requested
            if open_report:
                webbrowser.open(f"file://{os.path.abspath(report_path)}")
                webbrowser.open(f"file://{os.path.abspath(text_report_path)}")
        else:
            print("No data available for generating a report")
    
    def generate_test_video(self, output_path="input.mp4", duration=30, fps=30, resolution=(800, 600)):
        """Generate a test video with objects moving on a conveyor belt"""
        print(f"Generating test video: {output_path}")
        print(f"Duration: {duration} seconds, FPS: {fps}, Resolution: {resolution}")
        
        # Use subprocess to run the test video generation script
        import subprocess
        subprocess.run([
            "python", "generate_test_video.py", 
            "--output", output_path,
            "--duration", str(duration),
            "--fps", str(fps),
            "--width", str(resolution[0]),
            "--height", str(resolution[1])
        ])
        print(f"Test video generated: {output_path}")


def main():
    """Main function to run the integrated cap inspection system"""
    parser = argparse.ArgumentParser(description="Integrated Bottle Cap Inspection System with Reporting")
    
    # Video source options
    parser.add_argument('--video', type=str, default='input.mp4', help='Path to video file or camera index')
    parser.add_argument('--generate-video', action='store_true', help='Generate test video before running inspection')
    parser.add_argument('--video-duration', type=int, default=30, help='Duration of test video in seconds')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory for all results')
    parser.add_argument('--db-path', type=str, default='output/defect_database.db', help='Path to database file')
    parser.add_argument('--record-video', action='store_true', help='Record inspection session to video')
    
    # Reporting options
    parser.add_argument('--generate-report', action='store_true', help='Generate report after inspection')
    parser.add_argument('--open-report', action='store_true', help='Open report in browser')
    parser.add_argument('--dashboard', action='store_true', help='Generate dashboard after inspection')
    
    # YOLO options
    parser.add_argument('--use-yolo', action='store_true', help='Use YOLOv8 for object detection')
    parser.add_argument('--yolo-model', type=str, help='Path to custom YOLO model')
    parser.add_argument('--yolo-conf', type=float, default=0.25, help='YOLO confidence threshold')
    parser.add_argument('--train-yolo', action='store_true', help='Train a custom YOLO model before running')
    parser.add_argument('--data-yaml', type=str, help='Path to data.yaml file for YOLO training')
    
    args = parser.parse_args()
    
    # Check if the video source is a file or camera index
    if not args.video or args.video == 'input.mp4':
        video_source = 'media/test_input.mp4'  # Default to 'test_input.mp4' in the 'media' folder
    else:
        video_source = args.video
    
    # Generate test video if requested
    if args.generate_video:
        print("Generating test video...")
        # Create temporary inspector just for test video generation
        temp_inspector = IntegratedCapInspection(video_source=0, output_dir=args.output_dir, db_path=args.db_path)
        temp_inspector.generate_test_video(output_path=args.video, duration=args.video_duration)
        
        # Wait a moment to ensure video file is properly closed
        time.sleep(1)
        
        # Update video source to the generated video
        video_source = args.video
    
    # Train YOLO model if requested
    if args.train_yolo and args.data_yaml:
        print("Training custom YOLO model...")
        detector = YOLODetector()
        model_path = detector.train_custom_model(args.data_yaml)
        if model_path:
            args.yolo_model = model_path
    
    # Create and run the integrated inspection system
    inspector = IntegratedCapInspection(
        video_source=video_source, 
        output_dir=args.output_dir, 
        db_path=args.db_path,
        use_yolo=args.use_yolo,
        yolo_model=args.yolo_model,
        yolo_conf=args.yolo_conf
    )
    
    # Set up video recording if requested
    if args.record_video:
        inspector.setup_video_recording(os.path.join(args.output_dir, "inspection_recording.mp4"))
    
    # Run the inspection system
    inspector.run()
    
    # Generate report if requested
    if args.generate_report:
        print("Generating final report...")
        inspector.generate_report(open_report=args.open_report)
    
    # Generate dashboard if requested
    if args.dashboard:
        print("Generating dashboard...")
        data_manager = DefectDataManager(args.db_path)
        data_manager.visualizer.output_dir = os.path.join(args.output_dir, "reports")
        
        # Remove this import
        # from defect_analyzer import generate_dashboard  
        
        # Use the already imported generate_dashboard
        generate_dashboard(data_manager, days=30, open_report=args.open_report)
        
        data_manager.close()

if __name__ == "__main__":
    main()
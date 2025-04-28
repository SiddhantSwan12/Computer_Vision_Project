import cv2
import numpy as np
import random
import os
import argparse
import sys
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

def generate_test_video(output_path="input.mp4", duration=30, fps=30, resolution=(800, 600)):
    """
    Generate a test video with objects moving on a conveyor belt.
    
    Args:
        output_path (str): Path to save the output video
        duration (int): Duration of the video in seconds
        fps (int): Frames per second
        resolution (tuple): Video resolution (width, height)
    """
    print(f"Generating test video: {output_path}")
    print(f"Duration: {duration} seconds, FPS: {fps}, Resolution: {resolution}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, resolution)
    
    # Define conveyor belt parameters
    belt_y = resolution[1] // 2
    belt_height = resolution[1] // 3
    belt_speed = 3  # pixels per frame
    
    # List of possible object shapes and colors
    # Primarily use circle shape to simulate bottle caps
    shapes = ['circle', 'circle', 'circle', 'circle', 'square']
    colors = [
        (0, 0, 255),    # Red
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 255, 0)   # Cyan
    ]
    
    # Create objects
    objects = []
    
    # Total number of frames
    total_frames = duration * fps
    
    # Fixed vertical position for straight-line appearance
    fixed_y = belt_y  # Center of the belt
    
    # Generate one object every 30 frames on average
    for i in range(total_frames // 30):
        # Object appears at frame number
        frame_start = random.randint(0, total_frames - 1)
        
        # Object properties
        shape = random.choice(shapes)
        color = random.choice(colors)
        size = random.randint(40, 60)  # More consistent size
        
        # Initial position (start from left side)
        x = -size
        y = fixed_y - size // 2  # Center the object on the conveyor line
        
        # Is object defective?
        defect_type = random.choice(['none', 'none', 'none', 'color', 'scratch', 'misalignment'])
        
        objects.append({
            'shape': shape,
            'color': color,
            'size': size,
            'x': x,
            'y': y,
            'frame_start': frame_start,
            'defect': defect_type
        })
    
    # Generate frames
    for frame_idx in range(total_frames):
        # Create a frame
        frame = np.ones((resolution[1], resolution[0], 3), dtype=np.uint8) * 240
        
        # Draw conveyor belt
        cv2.rectangle(frame, 
                     (0, belt_y - belt_height // 2), 
                     (resolution[0], belt_y + belt_height // 2), 
                     (120, 120, 120), -1)
        
        # Draw belt lines
        for i in range(0, resolution[0], 50):
            offset = (frame_idx * belt_speed) % 50
            x = (i + offset) % resolution[0]
            cv2.line(frame, 
                    (x, belt_y - belt_height // 2), 
                    (x, belt_y + belt_height // 2), 
                    (100, 100, 100), 2)
        
        # Draw central ROI
        roi_width = resolution[0] // 5
        roi_height = belt_height // 2
        roi_x = (resolution[0] - roi_width) // 2
        roi_y = belt_y - roi_height // 2
        cv2.rectangle(frame, 
                     (roi_x, roi_y), 
                     (roi_x + roi_width, roi_y + roi_height), 
                     (50, 200, 50), 2)
        
        # Draw objects
        for obj in objects:
            if frame_idx >= obj['frame_start']:
                # Update object position
                obj['x'] += belt_speed
                
                # If object is still visible
                if obj['x'] < resolution[0]:
                    # Draw object
                    if obj['shape'] == 'circle':
                        # Draw a bottle cap-like object with gold ring and white/colored center
                        # Draw outer ring (gold color)
                        cv2.circle(frame, (obj['x'] + obj['size']//2, obj['y'] + obj['size']//2), 
                                  obj['size'] // 2, (0, 215, 255), -1)  # Gold color
                        
                        # Draw inner part (white or another color)
                        inner_color = (255, 255, 255)  # White by default
                        if obj['defect'] == 'color':
                            inner_color = (100, 100, 200)  # Different color for defect
                            
                        cv2.circle(frame, (obj['x'] + obj['size']//2, obj['y'] + obj['size']//2), 
                                  obj['size'] // 3, inner_color, -1)
                    
                    elif obj['shape'] == 'rectangle':
                        cv2.rectangle(frame, 
                                    (obj['x'], obj['y']), 
                                    (obj['x'] + obj['size'], obj['y'] + obj['size'] // 2), 
                                    obj['color'], -1)
                    
                    elif obj['shape'] == 'square':
                        cv2.rectangle(frame, 
                                    (obj['x'], obj['y']), 
                                    (obj['x'] + obj['size'], obj['y'] + obj['size']), 
                                    obj['color'], -1)
                    
                    elif obj['shape'] == 'triangle':
                        points = np.array([
                            [obj['x'], obj['y'] + obj['size']],
                            [obj['x'] + obj['size'], obj['y'] + obj['size']],
                            [obj['x'] + obj['size'] // 2, obj['y']]
                        ])
                        cv2.fillPoly(frame, [points], obj['color'])
                    
                    # Add defect to object
                    if obj['defect'] != 'none':
                        if obj['defect'] == 'scratch':
                            # Add a scratch (line)
                            center_x = obj['x'] + obj['size'] // 2
                            center_y = obj['y'] + obj['size'] // 2
                            x1 = center_x - obj['size'] // 4
                            y1 = center_y - obj['size'] // 4
                            x2 = center_x + obj['size'] // 4
                            y2 = center_y + obj['size'] // 4
                            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
                        
                        elif obj['defect'] == 'dent':
                            # Add a dent (small dark area)
                            center_x = obj['x'] + obj['size'] // 2
                            center_y = obj['y'] + obj['size'] // 2
                            cv2.circle(frame, (center_x, center_y), 5, (50, 50, 50), -1)
                        
                        elif obj['defect'] == 'color':
                            # Color variation already applied in main drawing
                            pass
                        
                        elif obj['defect'] == 'misalignment':
                            # Add misalignment (part of the object shifted)
                            if obj['shape'] == 'circle':
                                # Draw a slightly offset inner ring
                                cv2.circle(frame, 
                                         (obj['x'] + obj['size']//2 + 5, obj['y'] + obj['size']//2 + 5), 
                                         obj['size'] // 4, 
                                         (100, 100, 100), 
                                         -1)
                            else:
                                # For other shapes, add a misaligned rectangle
                                shift = obj['size'] // 3
                                cv2.rectangle(frame, 
                                            (obj['x'] + shift, obj['y'] + shift), 
                                            (obj['x'] + obj['size'], obj['y'] + obj['size']), 
                                            (0, 0, 0), 
                                            -1)
        
        # Add frame number
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Write frame to video
        out.write(frame)
        
        # Progress indicator (every 10%)
        if frame_idx % (total_frames // 10) == 0:
            progress = (frame_idx / total_frames) * 100
            print(f"Progress: {progress:.1f}%")
    
    # Release everything
    out.release()
    print(f"Video saved to {output_path}")
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a test video with bottle caps moving on a conveyor belt")
    parser.add_argument("--output", type=str, default="input.mp4", help="Output video path")
    parser.add_argument("--duration", type=int, default=30, help="Video duration in seconds")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--width", type=int, default=800, help="Video width")
    parser.add_argument("--height", type=int, default=600, help="Video height")
    
    args = parser.parse_args()
    resolution = (args.width, args.height)
    
    generate_test_video(args.output, args.duration, args.fps, resolution) 
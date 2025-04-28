#!/usr/bin/env python3
import os
import argparse
import subprocess
import time

def main():
    """Main function to run bottle cap inspection system"""
    parser = argparse.ArgumentParser(description="Run Bottle Cap Inspection System")
    # Existing arguments
    parser.add_argument("--skip-video-gen", action="store_true", help="Skip test video generation")
    parser.add_argument("--video-duration", type=int, default=30, help="Duration of test video in seconds")
    parser.add_argument("--video-output", type=str, default="input.mp4", help="Output path for test video")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory for inspection results")
    
    # New arguments for YOLO and reporting
    parser.add_argument("--use-yolo", action="store_true", help="Use YOLOv8 for object detection")
    parser.add_argument("--generate-report", action="store_true", help="Generate report after inspection")
    parser.add_argument("--dashboard", action="store_true", help="Generate dashboard after inspection")
    parser.add_argument("--open-report", action="store_true", help="Open report in browser")

    args = parser.parse_args()
    
    # Update the script to use 'test_input.mp4' as the default video input if no other video is specified
    if args.video_output == 'input.mp4':
        args.video_output = 'media/test_input.mp4'  # Default to 'test_input.mp4' in the 'media' folder
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Generate test video if not skipped
    if not args.skip_video_gen:
        print("Step 1: Generating test video...")
        subprocess.run([
            "python", "src/tools/video/generate_test_video.py", 
            "--output", args.video_output,
            "--duration", str(args.video_duration)
        ])
        print(f"Test video generated: {args.video_output}")
        
        # Wait a moment to ensure video file is properly closed
        time.sleep(1)
    else:
        print("Skipping video generation, using existing video file.")
    
    # Step 2: Run cap inspection UI
    print("Step 2: Starting Bottle Cap Inspection UI...")
    subprocess.run([
        "python", "src/ui/cap_inspection_ui.py",
        "--video", args.video_output,
        "--output", args.output_dir
    ])
    
    # Step 3: Run integrated cap inspection with additional arguments
    print("Step 3: Running Integrated Cap Inspection...")
    cmd = [
        "python", "src/detection/integrated_cap_inspection.py",
        "--video", args.video_output,
        "--output-dir", args.output_dir
    ]
    
    # Add optional arguments if specified
    if args.use_yolo:
        cmd.append("--use-yolo")
    if args.generate_report:
        cmd.append("--generate-report")
    if args.dashboard:
        cmd.append("--dashboard")
    if args.open_report:
        cmd.append("--open-report")
    
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
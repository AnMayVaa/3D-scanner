import pyrealsense2 as rs
import numpy as np
import cv2
import base64
import time
import json
import sys
import os
import threading

# --- Configuration ---
# Folders to store output files
RGB_FOLDER = "saved_images/RGB"
DEPTH_FOLDER = "saved_images/Depth"
PLY_FOLDER = "saved_images/PLY"

# Create directories if they don't exist
for folder in [RGB_FOLDER, DEPTH_FOLDER, PLY_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Global flag to trigger image capture
save_flag = False

# --- Helper Function: Listen for Node.js Commands ---
# This runs in a separate thread to prevent blocking the main camera loop
def listen_to_nodejs():
    global save_flag
    while True:
        try:
            # Read line from Standard Input (stdin) sent by Node.js
            line = sys.stdin.readline()
            if line:
                data = json.loads(line)
                # Check if the command is "save"
                if data.get("cmd") == "save":
                    save_flag = True
        except:
            pass # Ignore errors during reading

# Start the listener thread
input_thread = threading.Thread(target=listen_to_nodejs, daemon=True)
input_thread.start()

# --- RealSense Setup ---
pipeline = rs.pipeline()
config = rs.config()

# Enable Depth and Color streams (640x480 resolution at 30 FPS)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# **CRITICAL**: Create an align object.
# This aligns the depth frame to the color frame so pixels match perfectly.
align = rs.align(rs.stream.color)

# Create PointCloud object for 3D calculation
pc = rs.pointcloud()

def main():
    global save_flag
    try:
        # Start streaming
        pipeline.start(config)
        
        # Send a "ready" signal to Node.js
        print(json.dumps({"status": "ready"}), flush=True)

        while True:
            # 1. Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            
            # 2. Align depth frame to color frame
            aligned_frames = align.process(frames)
            
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are available
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays for OpenCV processing
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # --- CAPTURE LOGIC ---
            if save_flag:
                timestamp = int(time.time())
                
                # Notify Node.js that saving has started
                print(json.dumps({"info": "Processing 3D data..."}), flush=True)

                # A. Save RGB Image
                cv2.imwrite(f"{RGB_FOLDER}/rgb_{timestamp}.jpg", color_image)
                
                # B. Save Depth Map (Colorized for visualization)
                # Convert 16-bit depth to 8-bit color map
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                cv2.imwrite(f"{DEPTH_FOLDER}/depth_{timestamp}.png", depth_colormap)
                
                # C. Save 3D Point Cloud (.ply)
                # Map the color frame to the point cloud (Texture Mapping)
                pc.map_to(color_frame)
                # Calculate XYZ coordinates
                points = pc.calculate(depth_frame)
                # Export to .ply file
                ply_filename = f"{PLY_FOLDER}/model_{timestamp}.ply"
                points.export_to_ply(ply_filename, color_frame)
                
                # Send success event back to Node.js
                print(json.dumps({"event": "saved", "path": ply_filename}), flush=True)
                
                # Reset flag
                save_flag = False

            # --- PREVIEW STREAM LOGIC ---
            # Resize image to 50% to reduce bandwidth and latency
            small_frame = cv2.resize(color_image, (0,0), fx=0.5, fy=0.5) 
            
            # Encode image to JPEG (Memory buffer)
            _, buffer = cv2.imencode('.jpg', small_frame)
            
            # Convert JPEG buffer to Base64 string for Web display
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')

            # Get distance at the center of the frame
            dist = depth_frame.get_distance(320, 240)

            # Create JSON payload
            payload = {
                "distance": round(dist, 3), # Distance in meters
                "image": jpg_as_text        # Base64 Image
            }
            
            # Send data to Node.js (flush=True is required!)
            print(json.dumps(payload), flush=True)

    except Exception as e:
        # Report any errors to Node.js
        print(json.dumps({"error": str(e)}), flush=True)

    finally:
        # Stop the camera pipeline when exiting
        pipeline.stop()

if __name__ == "__main__":
    main()
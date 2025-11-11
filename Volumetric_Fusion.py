import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2
import time

# =============================================================================
# --- User Adjustable Settings (Tuned for FUSION) ---
# =============================================================================
# 1. Scanning distance
MIN_DISTANCE = 0.1
MAX_DISTANCE = 0.3

# 2. Model quality settings (TSDF)
VOXEL_LENGTH = 0.001 # 1mm (ความละเอียดของ "กล่อง" ที่ใช้หลอมโมเดล)
SDF_TRUNC = VOXEL_LENGTH * 5 # ค่ามาตรฐาน (ไม่ต้องเปลี่ยน)

# 3. Crop Box dimensions
CROP_X_MIN = -0.07
CROP_X_MAX = 0.07
CROP_Y_MIN = -0.08
CROP_Y_MAX = 0.08
CROP_Z_MIN = -0.06
CROP_Z_MAX = 0.06

# 4. Camera resolution
CAM_WIDTH = 640  # (ใช้ 640x480 จะเสถียรกว่า 1280x720 สำหรับการ FUSION ต่อเนื่อง)
CAM_HEIGHT = 480
CAM_FPS = 30
# =============================================================================


class FusionScanner:
    def __init__(self, width=CAM_WIDTH, height=CAM_HEIGHT, fps=CAM_FPS):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        
        # ฟิลเตอร์กรองระยะ
        self.depth_filter = rs.threshold_filter()
        self.depth_filter.set_option(rs.option.min_distance, MIN_DISTANCE)
        self.depth_filter.set_option(rs.option.max_distance, MAX_DISTANCE)

        # ฟิลเตอร์ลด Noise (สำคัญมากสำหรับ Fusion)
        self.spatial_filter = rs.spatial_filter(0.5, 20, 2, 0)
        self.temporal_filter = rs.temporal_filter(0.4, 20, 3)

        # --- นี่คือ "กล่อง" ที่ใช้หลอมโมเดล ---
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=VOXEL_LENGTH,
            sdf_trunc=SDF_TRUNC,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        
        self.is_running = True
        self.is_recording = False
        self.pinhole_camera_intrinsic = None
        
    def start(self):
        try:
            profile = self.pipeline.start(self.config)
        except RuntimeError as e:
            print(f"\n[ERROR] Could not start camera: {e}")
            return

        self.align = rs.align(rs.stream.color)
        intrinsics_rs = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        
        # แปลงค่า Intrinsics ของ RealSense เป็นของ Open3D
        self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            intrinsics_rs.width, intrinsics_rs.height, 
            intrinsics_rs.fx, intrinsics_rs.fy, 
            intrinsics_rs.ppx, intrinsics_rs.ppy)
            
        print("Camera started. Pinhole intrinsics configured.")
        self.run_scanner_loop()

    def stop(self):
        self.is_running = False
        if self.pipeline.get_active_profile():
            self.pipeline.stop()
        cv2.destroyAllWindows()

    def get_filtered_aligned_frames(self):
        try:
            frames = self.pipeline.wait_for_frames(1000) # Wait 1 second
        except RuntimeError:
            print("Timeout waiting for frames.")
            return None, None

        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            return None, None
        
        # กรองด้วยระยะทาง
        depth_frame = self.depth_filter.process(depth_frame)
        # กรองด้วยฟิลเตอร์ลด Noise
        depth_frame = self.spatial_filter.process(depth_frame)
        depth_frame = self.temporal_filter.process(depth_frame)
            
        return depth_frame, color_frame

    def run_scanner_loop(self):
        """Main loop: Press 's' to start/stop recording."""
        print("\n" + "="*50)
        print(" 3D Reconstruction Scanner is Ready!")
        print(" - **IMPORTANT:** Mount camera on a tripod (MUST be still).")
        print(" - Subject (person) should sit on a rotating chair.")
        print("\n - [S]: Start Recording")
        print("       (Then, slowly rotate the chair 180-360 degrees)")
        print(" - [Q]: Stop Recording & Process Model")
        print(" - [ESC]: Quit without processing")
        print("="*50 + "\n")
        
        frame_count = 0
        
        while self.is_running:
            depth_frame, color_frame = self.get_filtered_aligned_frames()
            if not depth_frame or not color_frame:
                continue
                
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # --- สร้างภาพ Preview ---
            preview_image = color_image.copy()
            
            # หาจุดที่อยู่ในระยะสแกนเพื่อไฮไลท์
            highlight_mask = (depth_image > (MIN_DISTANCE * 1000)) & (depth_image < (MAX_DISTANCE * 1000))
            overlay = preview_image.copy()
            overlay[highlight_mask] = [0, 255, 0] # BGR
            preview_image = cv2.addWeighted(preview_image, 0.7, overlay, 0.3, 0)
            
            # --- แสดงสถานะ ---
            if self.is_recording:
                instruction_text = f"RECORDING... Frame {frame_count}. Press 'q' to FINISH."
                cv2.putText(preview_image, instruction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # --- นี่คือหัวใจของการ "หลอม" ---
                rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)),
                    o3d.geometry.Image(depth_image),
                    convert_rgb_to_intensity=False)
                
                # เรา "หลอม" (integrate) ภาพนี้เข้าไปใน "กล่อง" (volume)
                # โดยบอกว่ากล้องไม่ได้ขยับ (np.identity(4))
                self.volume.integrate(
                    rgbd_image, 
                    self.pinhole_camera_intrinsic, 
                    np.identity(4) # <-- บอกว่ากล้องอยู่นิ่ง
                )
                frame_count += 1
                # --------------------------------
                
            else:
                instruction_text = "Ready. Press 's' to START recording."
                cv2.putText(preview_image, instruction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Live Preview (Green = Scanning Zone)", preview_image)
            key = cv2.waitKey(1)

            if key == ord('s') and not self.is_recording:
                self.is_recording = True
                frame_count = 0
                print("\nINFO: === STARTING RECORDING === (Slowly rotate the chair now)")

            elif key == ord('q') and self.is_recording:
                self.is_recording = False
                self.is_running = False # จบการทำงาน
                print(f"\nINFO: === STOPPING RECORDING === (Captured {frame_count} frames)")
                print("       Processing the final model... This may take a moment.")
                self.create_final_model() # ไปสร้างโมเดล

            elif key == 27: # ESC key
                self.is_running = False # ออกโดยไม่ทำอะไร
                print("\nINFO: Quitting program.")

        self.stop()

    def create_final_model(self, output_filename="ear_fusion_model.ply"):
        """Extracts the model from the volume, cleans it, and saves it."""
        print("INFO: 1. Extracting Point Cloud from TSDF volume...")
        pcd = self.volume.extract_point_cloud()
        
        if not pcd.has_points():
            print("ERROR: No points were extracted from the volume. Was the object in range?")
            return

        # Flip to be upright
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        print(f"INFO: Extracted {len(pcd.points)} points.")

        # --- 2. Crop ---
        print("INFO: 2. Cropping point cloud...")
        pcd_center = pcd.get_center()
        min_bound = pcd_center + np.array([CROP_X_MIN, CROP_Y_MIN, CROP_Z_MIN])
        max_bound = pcd_center + np.array([CROP_X_MAX, CROP_Y_MAX, CROP_Z_MAX])
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        pcd_cropped = pcd.crop(bbox)
        if not pcd_cropped.has_points():
            print("ERROR: Cropping removed all points. Check CROP settings.")
            pcd_cropped = pcd

        # --- 3. Clean (Outlier removal) ---
        print("INFO: 3. Removing statistical outliers...")
        pcd_cleaned, _ = pcd_cropped.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        if not pcd_cleaned.has_points():
            pcd_cleaned = pcd_cropped

        # --- 4. Create Mesh (Poisson) ---
        print("INFO: 4. Creating mesh using Poisson reconstruction...")
        pcd_cleaned.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_LENGTH * 2, max_nn=30))
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd_cleaned, depth=9)
        
        # --- 5. Clean Mesh ---
        print("INFO: 5. Cleaning up the mesh...")
        densities = np.asarray(densities)
        vertices_to_remove = densities < np.quantile(densities, 0.05)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
            triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        if len(cluster_n_triangles) > 0:
            largest_cluster_idx = cluster_n_triangles.argmax()
            triangles_to_remove = triangle_clusters != largest_cluster_idx
            mesh.remove_triangles_by_mask(triangles_to_remove)
            print(f"INFO: Kept the largest mesh cluster.")
        
        # --- 6. Save and Display ---
        print(f"INFO: 6. Saving final model to {output_filename}")
        o3d.io.write_triangle_mesh(output_filename, mesh, write_vertex_colors=True)
        
        print("\nSUCCESS! Model saved. Displaying final mesh...")
        o3d.visualization.draw_geometries([mesh], window_name="Final Fused Model")


if __name__ == "__main__":
    scanner = FusionScanner()
    try:
        scanner.start()
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        print("\nProgram finished.")
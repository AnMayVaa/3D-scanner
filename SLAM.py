import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2

# =============================================================================
# --- User Adjustable Settings (Tuned for SLAM/Odometry) ---
# =============================================================================
# 1. Scanning distance
MIN_DISTANCE = 0.1
MAX_DISTANCE = 0.3

# 2. Model quality settings
VOXEL_LENGTH = 0.001 # 1mm (ความละเอียดสูงสุด)
SDF_TRUNC = VOXEL_LENGTH * 5

# 3. Crop Box dimensions
CROP_X_MIN = -0.07
CROP_X_MAX = 0.07
CROP_Y_MIN = -0.08
CROP_Y_MAX = 0.08
CROP_Z_MIN = -0.06
CROP_Z_MAX = 0.06

# 4. Camera resolution
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS = 30

# 5. Live Preview Settings
PREVIEW_VOXEL_SIZE = 0.005 # (5mm) ยิ่งค่ามาก ยิ่งหยาบ แต่ยิ่งเร็ว
UPDATE_PREVIEW_EVERY_N_FRAMES = 10 # อัปเดตหน้าต่าง 3D ทุกๆ 10 เฟรม
# =============================================================================


class SlamScanner:
    def __init__(self, width=CAM_WIDTH, height=CAM_HEIGHT, fps=CAM_FPS):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        
        self.depth_filter = rs.threshold_filter(MIN_DISTANCE, MAX_DISTANCE)
        self.spatial_filter = rs.spatial_filter(0.5, 20, 2, 0)
        self.temporal_filter = rs.temporal_filter(0.4, 20, 3)

        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=VOXEL_LENGTH,
            sdf_trunc=SDF_TRUNC,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        
        self.is_running = True
        self.is_recording = False
        self.pinhole_camera_intrinsic = None
        
        # ตัวแปรสำหรับ Odometry (การติดตามกล้อง)
        self.current_camera_pose = np.identity(4) 
        self.previous_rgbd_image = None 
        
        # --- (ส่วนที่เพิ่มเข้ามา) ตัวแปรสำหรับหน้าต่าง 3D Live Preview ---
        self.vis = o3d.visualization.Visualizer()
        self.live_preview_frame_count = 0
        # -------------------------------------------------------------

    def start(self):
        try:
            profile = self.pipeline.start(self.config)
        except RuntimeError as e:
            print(f"\n[ERROR] Could not start camera: {e}")
            return

        self.align = rs.align(rs.stream.color)
        intrinsics_rs = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        
        self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            intrinsics_rs.width, intrinsics_rs.height, 
            intrinsics_rs.fx, intrinsics_rs.fy, 
            intrinsics_rs.ppx, intrinsics_rs.ppy)
            
        print("Camera started. Pinhole intrinsics configured.")
        
        # --- (ส่วนที่เพิ่มเข้ามา) สร้างหน้าต่าง 3D ---
        print("INFO: Creating live 3D preview window...")
        self.vis.create_window("Live 3D Scan Preview (Sparse)", width=800, height=600)
        self.vis.get_render_option().point_size = 1.0
        # ------------------------------------------
        
        self.run_scanner_loop()

    def stop(self):
        self.is_running = False
        if self.pipeline.get_active_profile():
            self.pipeline.stop()
        
        # --- (ส่วนที่เพิ่มเข้ามา) ปิดหน้าต่าง 3D ---
        self.vis.destroy_window()
        # ---------------------------------------
        
        cv2.destroyAllWindows()

    def get_filtered_rgbd_image(self):
        """รับเฟรม, กรอง Noise, และแปลงเป็น Open3D RGBDImage"""
        try:
            frames = self.pipeline.wait_for_frames(1000)
        except RuntimeError:
            return None, None
            
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None
        
        depth_frame = self.depth_filter.process(depth_frame)
        depth_frame = self.spatial_filter.process(depth_frame)
        depth_frame = self.temporal_filter.process(depth_frame)
            
        depth_image_o3d = o3d.geometry.Image(np.asanyarray(depth_frame.get_data()))
        color_image_o3d = o3d.geometry.Image(cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2RGB))
        
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_image_o3d, depth_image_o3d, convert_rgb_to_intensity=False)
            
        return rgbd_image, np.asanyarray(color_frame.get_data())

    def run_scanner_loop(self):
        print("\n" + "="*50)
        print(" 3D SLAM Scanner is Ready! (LIDAR App Style)")
        print(" - **IMPORTANT:** You must move the CAMERA (by hand).")
        print(" - The Subject (person) MUST be PERFECTLY STILL.")
        print("\n - [S]: Start Recording")
        print("       (Then, SLOWLY move the camera around the ear like 'painting')")
        print(" - [Q]: Stop Recording & Process Model")
        print(" - [ESC]: Quit without processing")
        print("="*50 + "\n")
        
        frame_count = 0
        
        odometry_option = o3d.pipelines.odometry.OdometryOption(
            depth_diff_max=0.01 
        )
        
        while self.is_running:
            current_rgbd_image, color_image_cv = self.get_filtered_rgbd_image()
            if not current_rgbd_image:
                continue
            
            preview_image = color_image_cv.copy()
            
            # --- แสดงสถานะ ---
            if self.is_recording:
                instruction_text = f"RECORDING... Frame {frame_count}. Press 'q' to FINISH."
                cv2.putText(preview_image, instruction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # --- นี่คือหัวใจของ SLAM ---
                if self.previous_rgbd_image is None:
                    print("INFO: First frame captured. Start moving SLOWLY.")
                else:
                    [success, trans, info] = o3d.pipelines.odometry.compute_rgbd_odometry(
                        current_rgbd_image, self.previous_rgbd_image,
                        self.pinhole_camera_intrinsic, np.identity(4), 
                        o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
                        odometry_option)
                    
                    if success:
                        self.current_camera_pose = np.dot(self.current_camera_pose, np.linalg.inv(trans))
                        
                        self.volume.integrate(
                            current_rgbd_image,
                            self.pinhole_camera_intrinsic,
                            self.current_camera_pose 
                        )
                        frame_count += 1
                        
                        # --- (ส่วนที่เพิ่มเข้ามา) อัปเดตหน้าต่าง 3D ---
                        self.live_preview_frame_count += 1
                        if self.live_preview_frame_count % UPDATE_PREVIEW_EVERY_N_FRAMES == 0:
                            
                            # 1. สร้าง PCD จากภาพปัจจุบัน
                            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                                current_rgbd_image, self.pinhole_camera_intrinsic)
                            
                            # 2. ย่อส่วน (Downsample) ให้หยาบๆ
                            pcd_sparse = pcd.voxel_down_sample(voxel_size=PREVIEW_VOXEL_SIZE)
                            
                            # 3. ย้ายตำแหน่งไปตามที่กล้องขยับ
                            pcd_sparse.transform(self.current_camera_pose)
                            
                            # 4. พลิกให้ตั้งตรง (เหมือนโมเดลสุดท้าย)
                            pcd_sparse.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

                            # 5. เพิ่มเข้าไปในหน้าต่าง 3D
                            self.vis.add_geometry(pcd_sparse, reset_bounding_box=False)
                        # ------------------------------------------

                    else:
                        print(f"WARNING: Tracking LOST! (Frame {frame_count}). Move slower or go back.")
                        cv2.putText(preview_image, "TRACKING LOST! MOVE SLOWER!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                self.previous_rgbd_image = current_rgbd_image
                
            else:
                instruction_text = "Ready. Point at ear. Press 's' to START."
                cv2.putText(preview_image, instruction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Live Preview (2D)", preview_image)
            
            # --- (ส่วนที่เพิ่มเข้ามา) ทำให้หน้าต่าง 3D ขยับได้ ---
            self.vis.poll_events()
            self.vis.update_renderer()
            # -------------------------------------------------
            
            key = cv2.waitKey(1)

            if key == ord('s') and not self.is_recording:
                self.is_recording = True
                frame_count = 0
                self.previous_rgbd_image = None
                self.current_camera_pose = np.identity(4)
                
                # --- (ส่วนที่เพิ่มเข้ามา) ล้างหน้าต่าง 3D เมื่อเริ่มใหม่ ---
                self.vis.clear_geometries()
                # -------------------------------------------------
                
                print("\nINFO: === STARTING RECORDING === (Move camera SLOWLY around the ear)")

            elif key == ord('q') and self.is_recording:
                self.is_recording = False
                self.is_running = False
                print(f"\nINFO: === STOPPING RECORDING === (Fused {frame_count} frames)")
                print("       Processing the final model...")
                self.create_final_model()

            elif key == 27:
                self.is_running = False
                print("\nINFO: Quitting program.")

        self.stop()

    def create_final_model(self, output_filename="ear_slam_model_final.ply"):
        """ส่วนนี้เหมือนเดิม 100% ครับ (ตัด, คลีน, ปะผุ, โชว์)"""
        print("INFO: 1. Extracting Point Cloud from TSDF volume...")
        pcd = self.volume.extract_point_cloud()
        if not pcd.has_points():
            print("ERROR: No points were extracted.")
            return

        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        print(f"INFO: Extracted {len(pcd.points)} points.")

        # --- 2. Crop ---
        print("INFO: 2. Cropping point cloud...")
        pcd_center = pcd.get_center()
        min_bound = pcd_center + np.array([CROP_X_MIN, CROP_Y_MIN, CROP_Z_MIN])
        max_bound = pcd_center + np.array([CROP_X_MAX, CROP_Y_MAX, CROP_Z_MAX])
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        pcd_cropped = pcd.crop(bbox)
        if not pcd_cropped.has_points(): pcd_cropped = pcd

        # --- 3. Clean ---
        print("INFO: 3. Removing statistical outliers...")
        pcd_cleaned, _ = pcd_cropped.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        if not pcd_cleaned.has_points(): pcd_cleaned = pcd_cropped

        # --- 4. Create Mesh ---
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
        
        # --- 6. Save and Display ---
        print(f"INFO: 6. Saving final model to {output_filename}")
        o3d.io.write_triangle_mesh(output_filename, mesh, write_vertex_colors=True)
        print("\nSUCCESS! Model saved. Displaying final mesh...")
        o3d.visualization.draw_geometries([mesh], window_name="Final SLAM Model")


if __name__ == "__main__":
    scanner = SlamScanner()
    try:
        scanner.start()
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        print("\nProgram finished.")
import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2

# =============================================================================
# --- User Adjustable Settings (Tuned for EAR Scanning) ---
# =============================================================================
# 1. Scanning distance
MIN_DISTANCE = 0.05
MAX_DISTANCE = 0.2

# 2. Model quality settings
VOXEL_SIZE = 0.001 # 1mm
ICP_FITNESS_THRESHOLD = 0.6 # เกณฑ์การยอมรับ (60%)

# 3. Crop Box dimensions
CROP_X_MIN = -0.07
CROP_X_MAX = 0.07
CROP_Y_MIN = -0.08
CROP_Y_MAX = 0.08
CROP_Z_MIN = -0.06
CROP_Z_MAX = 0.06

# 4. Poses and instructions
POSES = {
    "1_center": "Look straight at camera, capture ear center. Press 'c'.",
    "2_front": "Turn head SLIGHTLY left (to show front of ear). Press 'c'.",
    "3_back": "Turn head SLIGHTLY right (to show back of ear). Press 'c'.",
    "4.top": "Tilt head SLIGHTLY down (to show top of ear). Press 'c'.",
    "5.bottom": "Tilt head SLIGHTLY up (to show ear lobe). Press 'c'."
}
# =============================================================================


class EarScanner:
    def __init__(self, width=640, height=480, fps=30): # (ถ้า Error "Couldn't resolve requests" ให้ลดเป็น 848, 480)
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

        # --- (ปรับแก้ 1) เพิ่มฟิลเตอร์ลด Noise ของ RealSense ---
        self.spatial_filter = rs.spatial_filter()
        self.spatial_filter.set_option(rs.option.filter_magnitude, 2)
        self.spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
        
        self.temporal_filter = rs.temporal_filter()
        self.temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.4)
        # --------------------------------------------------------

        self.accumulated_pcd = o3d.geometry.PointCloud()
        self.accumulated_pcd_features = None 
        self.is_running = True
        self.scanning_started = False
        self.current_pose_index = 0
        self.pose_keys = sorted(POSES.keys())

    def start(self):
        try:
            profile = self.pipeline.start(self.config)
        except RuntimeError as e:
            print(f"\n[ERROR] Could not start camera: {e}")
            print("        - Is the camera connected to a USB 3.0 port?")
            print("        - Is another program (like RealSense Viewer) using the camera?")
            print("        - Try lowering the resolution (e.g., 848x480).")
            return

        self.align = rs.align(rs.stream.color)
        intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("Live 3D Model", width=800, height=600)
        self.vis.get_render_option().point_size = 1.0
        self.vis.get_render_option().show_coordinate_frame = True

        self.run_scanner_loop()

    def stop(self):
        self.is_running = False
        if self.pipeline.get_active_profile():
            self.pipeline.stop()
        self.vis.destroy_window()
        cv2.destroyAllWindows()

    # --- (ปรับแก้ 1) อัปเดตฟังก์ชันนี้ให้เรียกใช้ฟิลเตอร์ ---
    def get_aligned_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        # 1. กรองด้วยระยะทาง
        filtered_depth_frame = self.depth_filter.process(depth_frame)
        
        # 2. กรองด้วยฟิลเตอร์ลด Noise
        filtered_depth_frame = self.spatial_filter.process(filtered_depth_frame)
        filtered_depth_frame = self.temporal_filter.process(filtered_depth_frame)
            
        return filtered_depth_frame, color_frame
    # ----------------------------------------------------

    def create_pcd_from_frames(self, depth_frame, color_frame):
        depth_image = o3d.geometry.Image(np.asanyarray(depth_frame.get_data()))
        color_image = o3d.geometry.Image(cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2RGB))
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_image, depth_image, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, self.pinhole_camera_intrinsic)
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        return pcd

    def preprocess_pcd(self, pcd):
        pcd_down = pcd.voxel_down_sample(VOXEL_SIZE)
        pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_SIZE * 2, max_nn=30))
        return pcd_down

    def compute_features(self, pcd_down):
        radius_feature = VOXEL_SIZE * 5
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_fpfh

    # --- (ปรับแก้ 2) อัปเดตฟังก์ชันนี้ ---
    def register_and_merge(self, new_pcd):
        """
        Registers and merges using a 2-stage (Global + Local) approach.
        """
        if not self.accumulated_pcd.has_points():
            # สแกนแรก (ท่า 1)
            self.accumulated_pcd = self.preprocess_pcd(new_pcd)
            self.accumulated_pcd_features = self.compute_features(self.accumulated_pcd) 
            self.vis.add_geometry(self.accumulated_pcd)
            self.vis.reset_view_point(True)
            self.vis.poll_events()
            print("INFO: First scan (Anchor) captured as reference.")
            return True
        else:
            # สแกนท่าต่อไป (ท่า 2-5)
            source_down = self.preprocess_pcd(new_pcd)
            source_fpfh = self.compute_features(source_down)
            target_down = self.accumulated_pcd
            target_fpfh = self.accumulated_pcd_features

            print("INFO: 1. Running Global Registration (RANSAC)...")
            
            ransac_distance_threshold = VOXEL_SIZE * 1.5
            result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_down, target_down, source_fpfh, target_fpfh, True,
                ransac_distance_threshold,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                3, [
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(ransac_distance_threshold)
                ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))

            if result_ransac.fitness < 0.1:
                print(f"WARNING: Global Registration FAILED! (RANSAC fitness: {result_ransac.fitness:.4f}).")
                print("         Could not find any common features. Try to move slower or have more overlap.")
                return False

            print(f"INFO: 1. Global (RANSAC) successful. Fitness: {result_ransac.fitness:.4f}")
            print("INFO: 2. Running Local Registration (ICP) for fine-tuning...")

            # --- 4. ขั้นตอนที่ 2: "เก็บรายละเอียด" (Local Registration) ---
            
            # ### ปรับแก้ 2.1: เพิ่มความ "ผ่อนปรน" ให้ระยะห่าง (จาก 0.5 เป็น 1.5) ###
            icp_distance_threshold = VOXEL_SIZE * 1.5 
            
            result_icp = o3d.pipelines.registration.registration_icp(
                source_down, target_down, icp_distance_threshold,
                result_ransac.transformation, 
                
                # ### ปรับแก้ 2.2: เปลี่ยนกลับไปใช้ PointToPoint ที่ "ทน Noise" กว่า ###
                o3d.pipelines.registration.TransformationEstimationPointToPoint()) 

            # 5. ตรวจสอบว่า ICP ทำงานสำเร็จหรือไม่
            if result_icp.fitness > ICP_FITNESS_THRESHOLD:
                # สำเร็จ!
                source_down.transform(result_icp.transformation)
                old_pcd = self.accumulated_pcd
                self.accumulated_pcd = old_pcd + source_down
                self.accumulated_pcd = self.accumulated_pcd.voxel_down_sample(VOXEL_SIZE)
                self.accumulated_pcd.estimate_normals() 
                self.accumulated_pcd_features = self.compute_features(self.accumulated_pcd)
                self.vis.remove_geometry(old_pcd, reset_bounding_box=False)
                self.vis.add_geometry(self.accumulated_pcd, reset_bounding_box=False)
                
                print(f"INFO: 2. Local (ICP) successful! Fitness: {result_icp.fitness:.4f}")
                return True
            else:
                print(f"WARNING: Local Registration FAILED! (ICP fitness: {result_icp.fitness:.4f}).")
                print("         Global match was found, but fine-tuning failed. Try again.")
                return False
    # ----------------------------------------------------

    def run_scanner_loop(self):
        """Main loop - ส่วนนี้ตรรกะเหมือนเดิม ไม่ต้องแก้"""
        print("\n" + "="*50)
        print(" Ear 3D Scanner is Ready! (Smart Version 2)")
        print(" - The green area is the ONLY scanning zone (MUST be close).")
        print(" - Follow the on-screen instructions carefully.")
        print(" - [S]: Start scanning.")
        print(" - [C]: Capture current pose.")
        print(" - [Q] or [ESC]: Finish and create final model.")
        print("\n--- Scanning Tips for Best Results ---")
        print(" - **SUBJECT MUST BE PERFECTLY STILL** (like a statue).")
        print(" - **LIGHTING IS KEY:** Use bright, even light (Ring Light is best).")
        print(" - **SMALL MOVEMENTS:** Turn/tilt your head very slowly between shots.")
        print(" - **OVERLAP:** Ensure each new pose still shows parts of the original ear.")
        print(" - **HAIR:** Pin all hair away from the ear.")
        print("="*50 + "\n")

        while self.is_running:
            try:
                depth_frame, color_frame = self.get_aligned_frames()
            except RuntimeError as e:
                print(f"Waiting for frames... Error: {e}")
                continue
                
            if not depth_frame or not color_frame:
                continue
            
            color_image = np.asanyarray(color_frame.get_data())
            depth_data = np.asanyarray(depth_frame.get_data())
            
            highlight_mask = (depth_data > 0)
            overlay = color_image.copy()
            overlay[highlight_mask] = [0, 255, 0]
            preview_image = cv2.addWeighted(color_image, 0.7, overlay, 0.3, 0)
            
            if not self.scanning_started:
                instruction_text = "Ready! Press 's' to start."
            elif self.current_pose_index < len(self.pose_keys):
                pose_key = self.pose_keys[self.current_pose_index]
                instruction = POSES[pose_key]
                instruction_text = f"Step {self.current_pose_index + 1}/{len(POSES)}: {instruction}"
            else:
                instruction_text = "All scans complete! Press 'q' to process."

            cv2.putText(preview_image, instruction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(preview_image, instruction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Live Preview (Green = Scanning Zone)", preview_image)
            
            key = cv2.waitKey(1)

            if key == ord('s') and not self.scanning_started:
                self.scanning_started = True
                print("INFO: Scanning process started!")

            elif key == ord('c') and self.scanning_started:
                if self.current_pose_index < len(self.pose_keys):
                    print(f"\nCapturing pose: {self.pose_keys[self.current_pose_index]}...")
                    new_pcd = self.create_pcd_from_frames(depth_frame, color_frame)
                    
                    if not new_pcd.has_points():
                        print("WARNING: No valid 3D points captured. Try again.")
                    else:
                        success = self.register_and_merge(new_pcd)
                        if success:
                            self.current_pose_index += 1
                            print("INFO: Pose captured successfully. Moving to next pose.")
                        else:
                            print("INFO: Please recapture the current pose.")
                            
                else:
                    print("INFO: All poses captured. Press 'q' to create final model.")

            elif key == ord('q') or key == 27:
                self.stop()
                break

            self.vis.poll_events()
            self.vis.update_renderer()

    def create_final_model(self, output_filename="ear_scan_model_final.ply"):
        """Creates the final mesh - ส่วนนี้เหมือนเดิมครับ"""
        if not self.accumulated_pcd.has_points():
            print("ERROR: No point cloud data was accumulated.")
            return

        print("\nINFO: Starting final model creation process...")
        
        # 1. Crop
        print("INFO: Cropping point cloud...")
        pcd_center = self.accumulated_pcd.get_center()
        min_bound = pcd_center + np.array([CROP_X_MIN, CROP_Y_MIN, CROP_Z_MIN])
        max_bound = pcd_center + np.array([CROP_X_MAX, CROP_Y_MAX, CROP_Z_MAX])
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        pcd_cropped = self.accumulated_pcd.crop(bbox)
        
        if not pcd_cropped.has_points():
            print("ERROR: Cropping removed all points.")
            pcd_cropped = self.accumulated_pcd

        print(f"INFO: Points remaining after crop: {len(pcd_cropped.points)}")

        # 2. Remove noise
        print("INFO: Removing statistical outliers...")
        pcd_cleaned, _ = pcd_cropped.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        if not pcd_cleaned.has_points():
            pcd_cleaned = pcd_cropped

        # 3. Create mesh (Poisson)
        print("INFO: Creating mesh using Poisson reconstruction...")
        pcd_cleaned.estimate_normals()
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd_cleaned, depth=10)
        
        # 4. Clean mesh
        print("INFO: Cleaning up the mesh...")
        densities = np.asarray(densities)
        vertices_to_remove = densities < np.quantile(densities, 0.05)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        # 5. Keep largest cluster
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
            triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        
        if len(cluster_n_triangles) > 0:
            largest_cluster_idx = cluster_n_triangles.argmax()
            triangles_to_remove = triangle_clusters != largest_cluster_idx
            mesh.remove_triangles_by_mask(triangles_to_remove)
            print(f"INFO: Kept the largest mesh cluster.")
        
        # 6. Save and display
        print(f"INFO: Saving final model to {output_filename}")
        o3d.io.write_triangle_mesh(output_filename, mesh, write_vertex_colors=True)
        
        print("\nSUCCESS! Model saved. Displaying final mesh...")
        o3d.visualization.draw_geometries([mesh], window_name="Final Cleaned Model (Ear)")


if __name__ == "__main__":
    scanner = EarScanner()
    try:
        scanner.start()
        # หลังจากลูปจบ (กด 'q') ค่อยสร้างโมเดล
        scanner.create_final_model()
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        # ตรวจสอบว่า pipeline หยุดทำงานแล้วแน่ๆ
        if scanner.pipeline and scanner.pipeline.get_active_profile():
            scanner.pipeline.stop()
        print("\nProgram finished.")
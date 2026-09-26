# 3D Scanner — Intel RealSense D435

Python tools for capturing 3D models with an **Intel RealSense D435** depth camera, tuned for close-range scanning of small subjects such as a human ear (10–30 cm). Each script outputs a cleaned `.ply` mesh built with Open3D.

The repo also includes a desktop capture app (Electron + Node.js + Python) in [`js/`](js/) — see [`js/Readme.md`](js/Readme.md).

## Scanning methods

| Script | Method | How to scan | Output |
|---|---|---|---|
| `Multi-view_Registration.py` | Captures 5 guided poses (center, front, back, top, bottom) and aligns them with RANSAC global registration + ICP refinement | Subject turns/tilts their head between shots | `ear_scan_model_final.ply` |
| `Volumetric_Fusion.py` | Continuous TSDF volume fusion | Camera fixed on a tripod, subject slowly rotates on a chair | `ear_fusion_model.ply` |
| `SLAM.py` | RGB-D odometry + TSDF fusion with a live 3D preview | Subject stays still, camera is moved slowly around them by hand | `ear_slam_model_final.ply` |

All three run the same post-processing: crop box → statistical outlier removal → Poisson surface reconstruction → mesh cleanup.

### Controls

- `S` — start scanning / recording
- `C` — capture the current pose (Multi-view only)
- `Q` — stop and build the final model
- `Esc` — quit

## Requirements

- Intel RealSense D435 on a **USB 3.0** port
- Python 3.8+

```bash
pip install pyrealsense2 open3d opencv-python numpy
python Volumetric_Fusion.py
```

## Tuning

Settings live at the top of each script:

- `MIN_DISTANCE` / `MAX_DISTANCE`: the depth range in meters.
- `VOXEL_SIZE` / `VOXEL_LENGTH`: the model resolution. The default is 1 mm.
- `CROP_*`: the bounding box around the subject.
- Camera resolution and FPS. If the camera fails to start, try 848×480.

### Tips for good scans

- Keep the subject completely still.
- Use bright, even lighting. A ring light works best.
- Move slowly and keep plenty of overlap between views.

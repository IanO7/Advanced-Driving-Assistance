## 🚗 Collision Avoidance (Depth-Based Object Detection) Methodology

---

## 📝 Overview
This project implements a real-time advanced driving assistance system (ADAS) by using MiDaS monocular depth estimation and YOLO object detection to trigger alerts for close objects (person, car, bus) based on the colorized depth map. 

---

## 🎬 Demo

<!-- TODO: Add demo images/videos here -->

---

## 🧠 Methodology

### 1. Depth Map Generation
- Uses a pre-trained MiDaS model to generate a per-pixel depth map from a single camera frame, capturing the scene's 3D structure.

### 2. Object Detection
- Uses YOLO in `depth_estimation.py` to detect cars, pedestrians, and buses in each frame, assigning bounding boxes and class labels.

### 3. Object Awareness and Depth-Aware Alerting (Green vs Red Icons)

- For each detected object, the system uses a two-stage logic (awareness and collision risk) as implemented in `depth_estimation.py`:
  - **Stage 1: Awareness (Green Icon):** A green icon shows presence of a detected object, with no depth check or alert.
  - **Stage 2: Collision Risk (Red Icon):** If over 75% of an object's box is close in the depth map, a red icon and alert are triggered, always overriding green.

> **Note:** The 75% threshold is applied to the bounding box (square/rectangle), but since the depth map closely matches object shapes, this balances catching close objects while minimizing false positives from background pixels (not object itself, but within bounding box).

### 4. Lane Departure Warning (LDW): 
- Based on `ldw.py`, this step detects lane lines and highlights the drivable area using edge detection and Hough transform overlays.

### 5. Visualization
- The original frame and colorized depth map (with bounding boxes and colorbar) are shown side by side for clarity.
- Alerts appear directly on the video feed for instant feedback.
- A yellow guide line, 10% from the bottom, helps align the bonnet and is only visible in the live display.
- The BirdsEyeView window has a Sensitivity slider to set how close an object must be for a red alert, with 58 as the recommended default.
- Alerts indicate Left, Center, or Right zones (not lane-based) for detected objects.

> **Note:** Objects fully below the yellow line (bonnet area) are ignored for alerts; only those above or touching the line are considered.

> **Best practice:** Add a bonnet overlay below the yellow line to show this area is not checked for alerts, avoiding confusion.

## 🕳️ Why Depth Map Instead of Pixel-Based (2D) Approaches?

**Contextual Awareness:** 2D pixel methods only estimate image distance, not real-world distance, and are affected by zoom and perspective, missing true depth.
**Relative 3D Structure:** Depth maps better capture which objects are closer or farther in the scene, even if zoom or angle changes, making alerts more meaningful.
**Robustness:** Combining object detection with depth estimation reduces false alerts and improves safety by focusing on true collision risks.

## ✨ Features
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Real--Time-green?logo=opencv&logoColor=white)
![MiDaS](https://img.shields.io/badge/Depth--Estimation-MiDaS-informational)
![YOLO](https://img.shields.io/badge/Object%20Detection-YOLOv11-orange)
![Bird's Eye View](https://img.shields.io/badge/Bird's%20Eye%20View-Visualization-blueviolet)
![Audio Alerts](https://img.shields.io/badge/Audio%20Alerts-Enabled-yellow)
![Lane Detection](https://img.shields.io/badge/Lane%20Detection-Edge%20%2B%20Hough-lightgrey)
![Modular Code](https://img.shields.io/badge/Modular-Extensible%20Python-9cf)
![Offline Support](https://img.shields.io/badge/Offline--Support-Yes-brightgreen)
![Accessible UI](https://img.shields.io/badge/Accessible%20UI-inclusive-yellowgreen)

## 🚦 Getting Started

### Anaconda Navigator GUI Method

1. In Anaconda Navigator, create a new environment (e.g., `adas_project`).
2. Launch the Anaconda Prompt from Navigator (not VS Code or standard terminal).
3. Run:
  ```powershell
  conda activate adas_project
  cd C:\project_folder_destination
  pip install -r requirements.txt
  ```
4. Done. Always use the Anaconda Prompt for running and installing, VS Code terminal won't work.

### VS Code Terminal CLI Method

### 1. Create a Python virtual environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

## 4. Run the system 
Use the main controller to run and configure features:
```powershell
python main.py
```

### Calibration
- Click "Calibrate" or press `c` in BirdsEyeView to auto-set Sensitivity using the nearest detected object; 58 is the recommended default.
- Adjust the slider to set how close an object must be for an alert (0=far, 255=near; pixels are "close" if index ≥ Sensitivity).

### Simple toggles (no-args Run button)
Running with no arguments uses defaults in `main.py`:
- `USE_CAMERA_DEFAULT`: webcam or video file
- `CAMERA_INDEX_DEFAULT`: webcam index
- `VIDEO_PATH_DEFAULT`: default video path
- `ENABLE_DEPTH_DEFAULT` / `ENABLE_LDW_DEFAULT`: enabled features
- `USE_IP_STREAM_DEFAULT` / `IP_STREAM_URL_DEFAULT`: auto-use live phone/IP stream

### Display & Alert Toggles
You can quickly enable/disable UI surfaces in `main.py`:
- `SHOW_MAIN_WINDOW_DEFAULT`: show/hide main window
- `SHOW_BIRDSEYE_DEFAULT`: show/hide Bird's Eye window
- `ALERT_SOUND_ENABLED_DEFAULT`: keep audio alerts even if windows are hidden
CLI flags always override these toggles.

### Command-line options (override toggles)
- Run with a video file:
```powershell
python main.py --video "test_videos/california_drive.mp4" --output output.mp4
```

- Run with webcam (index 0):
```powershell
python main.py --camera 0 --output output.mp4
```
IP / phone stream (explicit command):
```powershell
& C:/Users/Admin/.conda/envs/adas_project/python.exe C:/Users/Admin/Desktop/Advanced-Driving-Assistance/main.py --depth --video "http://10.211.119.11:8080/video"
```
IP / phone stream (toggle only): set `USE_IP_STREAM_DEFAULT = True`, then:
```powershell
python main.py
```


> **Note:** Do not run `depth_estimation.py` or `ldw.py` directly. Use `main.py` to control all features.

## 📝 Notes
- For research/prototyping; optimize and test before deployment.
- Methodology can extend to more object classes or sensors.
- Detects cars, pedestrians, buses, and trucks only.
- Bikes/motorcycles excluded; riders detected as pedestrians.
- Multi-zone logic works for all roads; more alerts in crowded/single-lane scenarios are expected.
- Latest-frame capture reduces buffer lag but not processing latency; very brief objects may be missed.

### ⚡ Parallel Processing Mode (Fast Path)
Reduces alert latency without changing collision logic:
- Detection and depth run in parallel threads (DetectionWorker, DepthWorker)
- Always uses the latest frame (no backlog)
- Optional smaller YOLO input (default 256) for higher FPS
- Adjustable detection/depth intervals (`--detection-interval`, `--depth-interval`)
- 2-stage logic, red-over-green priority, and 75% close-pixel check unchanged
- Inferno colormap, sensitivity default (58), and box proximity logic unchanged
- Trade-offs: box tightness may vary with smaller input; very fast objects may have minor detection/depth mismatch
- To disable: run with `--no-parallel --detection-imgsz 288`


## 📄 License
See [LICENSE](LICENSE).

## 📁 Repository Contents

- `main.py` — Main controller: enables/disables features (depth, LDW, bird’s eye view)
- `depth_estimation.py` — Core logic: depth estimation, object detection, alerting, bird’s eye view visualization
- `ldw.py` — Lane detection and overlay logic (modular LDW)
- `requirements.txt` — Python dependencies
- `LICENSE` — License file
- `yolo11n.pt` — YOLOv11 weights
- `assets/` — Icons and overlay images (e.g., `green_car.png`, `red_person.png`, `birds_eye_view_car.png`, `alert_sound.mp3`)
- `test_videos/` — Example/test videos (e.g., `california_drive.mp4`, `car_crash.mp4`, `pedestrian_crash.mp4`, `depth_video.mp4`, `japan_drive.mp4`, `output.mp4`)
- `output.mp4` — Example output video
- `example_good.mp4` — Example good run video
- `__pycache__/` — Python cache files
- `.vscode/` — VS Code settings
- `.gitignore` — Git ignore file
- `.git/` — Git repository metadata

---

## 🔗 References

This project includes small portions and ideas inspired by:
- MiDaS (Intel ISL): https://github.com/isl-org/MiDaS
- Lane Detection (reference implementation): https://github.com/maheshpaulj/Lane_Detection
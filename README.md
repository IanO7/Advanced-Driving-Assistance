## Collision Avoidance (Depth-Based Object Detection) Methodology

This project uses MiDaS depth estimation and YOLO object detection to trigger alerts for close objects (person, car, bus) based on the colorized depth map.

---

## Overview
This project implements a real-time advanced driving assistance system (ADAS) that combines monocular depth estimation with object detection to provide robust collision and pedestrian alerts. The system leverages deep learning models for both depth mapping and object recognition, enabling reliable safety warnings.

---

## Methodology

### 1. Depth Map Generation
- Uses a pre-trained MiDaS model (Intel Labs) to estimate a dense depth map from a single camera frame.
- The depth map provides per-pixel distance information, allowing the system to understand the 3D structure of the scene.

### 2. Object Detection
- Based on `depth_estimation.py`. Utilizes a YOLO (You Only Look Once) model to detect objects such as cars, pedestrians, and buses in each frame.
- Each detected object is assigned a bounding box and a class label.

### 3. Object Awareness and Depth-Aware Alerting (Green vs Red Icons)

- Based on `depth_estimation.py`. For each detected object, the system follows a two-stage logic (BUT CODE PROCESS IS ALWAYS RED THEN GREEN FOR ALL DRAWING IN MAP/LHS/VISUAL CASES CORRECT TO ACHIEVE BELOW WHEN PRESENTED/VISUALISED):
  - **Stage 1: Awareness (Green Icon)**
    - If an object (car, bus, truck, or pedestrian) is detected by YOLO/OpenCV, a green icon is shown in the bird’s eye view for that zone. This indicates presence only—no distance or depth check is performed, and no alert sound or warning is triggered.
  - **Stage 2: Collision Risk (Red Icon)**
    - The system examines the corresponding region in the depth map for each detected object. The Inferno colormap is used to visualize depth; yellow/white (indices 58–255 determined experimentally) represents the closest regions. If more than 75% of the pixels in the object's bounding box match this close-range color (empirically determined), a red icon is shown (alert), and a visual and audio warning is triggered.
    - The red icon always takes priority over green in a given zone.

**Note:** The 75% threshold is calculated over the bounding box, but the depth map itself is quite accurate to the true outline and waviness of the object (like a person or car). This means the box may include some background or pixels of different color/depth, since the box is general but the depth map color closely follows the object's shape. The threshold is chosen to balance catching most close objects while avoiding false positives from background pixels inside the box. 

### 4. Lane Departure Warning (LDW): 
- Based on `ldw.py`. Detects lane lines in the video frame using edge detection, region masking, and Hough transform, then overlays the detected lanes and highlights the drivable area on the original image.

### 5. Visualization
- The original frame and the colorized depth map (with bounding boxes and a colorbar) are displayed side by side for intuitive understanding.
- Alerts are overlaid on the video feed for immediate feedback.
- A horizontal yellow guide line is drawn 10% from the bottom of the live display to help you align the vehicle's bonnet (hood) just below it. This improves real-world depth estimation. The line is only visible in the live display, not in saved videos, and is not part of the actual video (just output).
- In the BirdsEyeView window, a "Sensitivity" slider lets you adjust how close an object must be to trigger a red alert icon. Lower values are more sensitive (alert for further objects), higher values require objects to be closer. The default is 58 (recommended), and the UI shows "(recommended)" if set to this value. Adjust live as needed.
- Alerts use Left / Center / Right: three equal-width forward sectors (not lane-based) indicating where an object lies.

**Note:** Any detected object whose bounding box is completely below this yellow line (i.e., in the bonnet area) is excluded from all alert and awareness logic. Only objects at least partially above or touching the line are considered for alerts or icons.

**Best practice:** For maximum clarity, fill the rectangle under the yellow line with a bonnet overlay (e.g., a gray or car-shaped region). This visually indicates that the area is not scanned for objects, even though depth is still estimated there, and prevents confusion about undetected objects in that region.

**Bird’s Eye View Icon Logic:**
- **Red icon:** Object detected and depth/collision risk confirmed (alert, with sound and visual warning).
- **Green icon:** Object detected by YOLO/OpenCV, but distance not checked—just shows presence (no sound, no alert, only visual awareness).

## Why Depth Map Instead of Pixel-Based (2D) Approaches?

- **Contextual Awareness:** Pixel-based (2D) methods only measure the distance between objects in image pixels, not their true real-world distance from the vehicle. This 2D pixel estimation is affected by camera zoom and perspective—if you zoom in, the pixel distance changes even if the real distance does not. It does not account for depth, so objects may appear close in the image but be far away in reality, or vice versa.
- **Relative 3D Structure:** Depth maps do not provide exact real-world distances either, and zoom can still affect their accuracy, but they capture the relative 3D structure of the scene much better than 2D pixel estimation. This means the system can tell which objects are closer or farther away within the same image, even if zoom or camera angle changes. For example, if object X is closer than object Y in the image, the depth map will reflect that relationship, making alerts more meaningful and robust.
- **Robustness:** By combining object detection with depth estimation, the system can more reliably identify true collision risks, reducing unnecessary alerts and improving safety.

## Features
- Real-time video processing with OpenCV
- Monocular depth estimation using MiDaS
- Object detection with YOLO
- Depth-aware alerting for cars, pedestrians, buses, and trucks
- Visual and audio warnings
- Lane detection (lane departure warning alert to be integrated later)
- **Bird’s Eye View visualization** with icon-based alerts for cars, pedestrians, buses, and trucks
- Modular, extensible Python code

## Getting Started

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

### Simple toggles (no-args Run button)
If you just press Run (F5) in VS Code or run `python main.py` with no arguments, the defaults are controlled at the top of `main.py`:
- `USE_CAMERA_DEFAULT`: set `True` to default to webcam, `False` to default to a video file
- `CAMERA_INDEX_DEFAULT`: which webcam index to use (usually `0`)
- `VIDEO_PATH_DEFAULT`: default MP4 path (e.g., `test_videos/california_drive.mp4`)
- `ENABLE_DEPTH_DEFAULT` / `ENABLE_LDW_DEFAULT`: which features are enabled when no flags are passed

### Display & Alert Toggles
Also in `main.py` you can quickly enable/disable UI surfaces without CLI flags:
- `SHOW_MAIN_WINDOW_DEFAULT`: show/hide combined Lane + Depth window.
- `SHOW_BIRDSEYE_DEFAULT`: show/hide Bird's Eye proximity/sensitivity window.
- `ALERT_SOUND_ENABLED_DEFAULT`: keep audio alerts even if all windows are hidden.

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

Tip: `--video 0` is also treated as webcam index `0`.

Do not run `depth_estimation.py` or `ldw.py` directly. Use `main.py` to control all features.

## Notes
- This system is designed for research and prototyping. For deployment, further optimization and testing are recommended.
- The methodology can be extended to other object classes or integrated with additional sensors for enhanced reliability.
- The system detects cars, pedestrians, buses, and trucks.
- Bicycles and motorcycles are not included, as stationary bikes/motorcycles are not a collision risk and any person present (cyclist/motorcyclist) would already be detected as a pedestrian.
- This system assumes a typical multi-lane road and does not restrict collision alerts to detected lanes. In single-lane or crowded town scenarios, more alerts may occur, which is expected due to closer objects. Lane-based alert restriction is not applied, as the multi-zone logic is designed for all types of roads.


**NEXT STEPS
- cotrol interface for threshold (pixel no need), and features to turn on
-calibration of sensitivity, 75%ok general, but which depth to alert close is sensitivity value to calibrate beforehand! size !=  matter since depth map clearly outlines accurately


## License
See [LICENSE](LICENSE).

## 📁 Repository Contents

- `main.py` — Main controller: enables/disables features (depth, LDW, bird’s eye view)
- `depth_estimation.py` — Core logic: depth estimation, object detection, alerting, bird’s eye view visualization
- `ldw.py` — Lane detection and overlay logic (modular LDW)
- `yolo11n.pt` — YOLOv11 weights

---

## 🛠 Requirements

- Python 3.8+
- Install dependencies:
  ```bash
  pip install opencv-python numpy ultralytics
  ```

---

## 🎨 Color Coding

**Bird’s Eye View Icons:**
- <span style="color:red">Red car/bus/person icon</span>: Alert (collision risk, depth checked, sound and visual warning)
- <span style="color:green">Green car/bus/person icon</span>: Object detected (YOLO/OpenCV), but depth not checked—shows presence only (no sound, no alert)
- No icon: Safe (no alert or detection in zone)

**Bounding box overlays:**
- <span style="color:red">Red</span>: Alert (car, bus, truck, or pedestrian)
- <span style="color:green">Green</span>: Safe car
- <span style="color:yellow">Yellow</span>: Guide line for bonnet alignment

---

## References

This project includes small portions and ideas inspired by:
- MiDaS (Intel ISL): https://github.com/isl-org/MiDaS
- Lane Detection (reference implementation): https://github.com/maheshpaulj/Lane_Detection
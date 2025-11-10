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
- Utilizes a YOLO (You Only Look Once) model to detect objects such as cars, pedestrians, and buses in each frame.
- Each detected object is assigned a bounding box and a class label.

### 3. Depth-Aware Alerting (Combining MiDaS Depth Map and YOLO Object Detection)
- For each detected object, the system examines the corresponding region in the depth map.
- The Inferno colormap is used to visualize depth; yellow/white (indices 58–255) represents the closest regions. If more than 75% of the pixels in the object's bounding box match this close-range color (empirically determined), an alert is triggered (visual and audio).

**Note:** The 75% threshold is calculated over the bounding box, but the depth map itself is quite accurate to the true outline and waviness of the object (like a person or car). This means the box may include some background or pixels of different color/depth, since the box is general but the depth map color closely follows the object's shape. The threshold is chosen to balance catching most close objects while avoiding false positives from background pixels inside the box. 

### 4. Visualization
- The original frame and the colorized depth map (with bounding boxes and a colorbar) are displayed side by side for intuitive understanding.
- Alerts are overlaid on the video feed for immediate feedback.
- A horizontal yellow guide line is drawn 10% from the bottom of the live display to help you align the vehicle's bonnet (hood) just below it. This improves real-world depth estimation. The line is only visible in the live display, not in saved videos, and is not part of the actual video (just output).

## Why Depth Map Instead of Pixel-Based (2D) Approaches?

- **Contextual Awareness:** Pixel-based (2D) methods only measure the distance between objects in image pixels, not their true real-world distance from the vehicle. This 2D pixel estimation is affected by camera zoom and perspective—if you zoom in, the pixel distance changes even if the real distance does not. It does not account for depth, so objects may appear close in the image but be far away in reality, or vice versa.
- **Relative 3D Structure:** Depth maps do not provide exact real-world distances either, and zoom can still affect their accuracy, but they capture the relative 3D structure of the scene much better than 2D pixel estimation. This means the system can tell which objects are closer or farther away within the same image, even if zoom or camera angle changes. For example, if object X is closer than object Y in the image, the depth map will reflect that relationship, making alerts more meaningful and robust.
- **Robustness:** By combining object detection with depth estimation, the system can more reliably identify true collision risks, reducing unnecessary alerts and improving safety.

## Features
- Real-time video processing with OpenCV
- Monocular depth estimation using MiDaS
- Object detection with YOLO
- Depth-aware alerting for cars, pedestrians, and buses
- Visual and audio warnings
- Modular, extensible Python code

## Getting Started

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

### 3. Download model weights
Ensure `yolo11n.pt` (YOLO weights) and the MiDaS model weights are present in the project directory. See Notes below if you need help downloading them.

### 4. Run depth estimation
Process your video file (e.g., `pedestrian_crash.mp4`) with:
```bash
python depth_estimation.py
```
By default, the script processes `pedestrian_crash.mp4` and outputs `depth_video.mp4`. To use a different video, edit the script's last line.

## Notes
- This system is designed for research and prototyping. For deployment, further optimization and testing are recommended.
- The methodology can be extended to other object classes or integrated with additional sensors for enhanced reliability.
- 
- **Next Steps:**
  -ENSURE BELOW LINE NOT CONSIDERED, STILL RELEVANT FOR DEPTH BUT REPRESENTS CAR SO DONT IDENIFY AS OBJECT THEN WONT ALERT
  - Integrate lane departure warning (LDW) to alert if the vehicle drifts out of its lane, PERFECT FEATURE FIRST THEN CAN ADD METHODOLOGY TO README LATER ON
  - If lane detection is effective, restrict collision alerts to only objects within the detected lane, reducing false positives from adjacent lanes.

## License
See [LICENSE](LICENSE).

## 📁 Repository Contents

- `pixel_estimation.py` — Main script: runs YOLO detection, draws lane overlays, and raises alerts
- `edge_detection.py` — Lane detection helper functions (thresholding, Sobel, blur)
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

- **Car detection:**
  - <span style="color:red">Red</span>: Car within collision threshold (danger)
  - <span style="color:green">Green</span>: Safe car
- **Pedestrian detection:**
  - <span style="color:magenta">Magenta</span>: Pedestrian alert (close)
  - <span style="color:cyan">Cyan</span>: Safe pedestrian

---

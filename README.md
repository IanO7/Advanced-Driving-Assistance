
<h1 align="center">🚗 Advanced Driving Assistance</h1>

<p align="center">
<b>YOLO-based real-time collision and pedestrian alert system with lane detection overlays.</b>
</p>

---

## Features

- **YOLOv11 object detection** for cars and pedestrians
- **Lane detection** using OpenCV (Sobel, Gaussian blur, thresholding)
- **Pixel-based distance estimation** for collision and pedestrian alerts
- **Color-coded bounding boxes** for easy visual feedback

---

## ⚠️ Calibration & Pixel Thresholds

> **Important:**
> - If you change the camera zoom or field of view, you must recalibrate the pixel thresholds for collision and pedestrian alerts.
> - The same pixel values do **not** represent the same real-world distance if the zoom changes, even if the video resolution stays the same.
> - For best results, use unscaled 1920x1080 video and avoid digital zoom.

---

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

## 🚀 Usage

1. Ensure dependencies are installed (see above)
2. Run the main script:
   ```bash
   python pixel_estimation.py
   ```
3. Adjust thresholds in `pixel_estimation.py` as needed (see below)

---

## ⚙️ Configuration

- **Collision threshold**: `COLLISION_THRESHOLD` (pixels)
- **Pedestrian threshold**: `PEDESTRIAN_THRESHOLD` (pixels)
- Both are defined in `pixel_estimation.py`
- Larger values = Larger threshold distance range and so more sensitive (alerts at greater distance)
- Thresholds are measured from the bottom-center reference point (red dot), ensure bonnet edge is at red dot
-**THRESHOLD STILL NEED TO CONFIGURE, 

---

## 🎨 Color Coding

- **Car detection:**
  - <span style="color:red">Red</span>: Car within collision threshold (danger)
  - <span style="color:green">Green</span>: Safe car
- **Pedestrian detection:**
  - <span style="color:magenta">Magenta</span>: Pedestrian alert (close)
  - <span style="color:cyan">Cyan</span>: Safe pedestrian

---



## 📝 Notes & Limitations

- Pixel-based distances are **not** real-world meters; always recalibrate if camera/lens/zoom changes
- For best accuracy, keep the camera fixed and avoid digital zoom
- Lane detection uses OpenCV pipeline (see `edge_detection.py`)
- **Distance measurement:**
  - In the original approach, OpenCV is used to detect objects and measure the pixel distance between their bounding boxes (object-to-object). This is simply a measurement of pixels on the screen, not real-world distance.
  - In this implementation, the reference is a fixed dot on the screen (bottom-center, usually at the bonnet edge). No OpenCV detection is needed for the reference point—only for the other objects. Pixel distances are measured from this fixed dot to the **closest edge** of each detected object's bounding box (not the centroid), using the same pixel logic as the original method.
  - Both methods measure pixel distances on the screen, but this approach simplifies the reference to a fixed location, making the logic more consistent and easier to calibrate.
  - **Optional:** If you click on a detected object in the display window, the script will show the vertical pixel distance from that object to all others (for visualization only, not for alerts).
- **Alert sound:** When a collision or pedestrian alert is triggered, an alert sound (`alert_sound.mp3`) will play (if present in the script directory). The sound will not overlap if multiple alerts occur in quick succession.


---

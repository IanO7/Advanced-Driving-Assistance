## Alert Methodology

This project uses MiDaS depth estimation and YOLO object detection to trigger alerts for close objects (person, car, bus) based on the colorized depth map.

**Alert logic:**
- The Inferno colormap is used to visualize depth. In this mapping, yellow/white (high values, 58–255) represents the closest regions.
- For each detected object, the percentage of pixels in its bounding box that match the yellow/white Inferno range (BGR values for indices 58–255) is calculated.
- An alert is triggered if more than 75% of the pixels in the box match this range.

**Why 58–255 and 75%?**
- These values were empirically determined to best match close (yellow) regions and robustly trigger alerts for objects that are actually close, while avoiding false positives.

**General methodology:**
1. Detect objects (person, car, bus) with YOLO.
2. For each object, extract the bounding box region from the colorized depth map.
3. Count the percentage of pixels in the box that match the allowed yellow/white Inferno range (58–255).
4. Trigger an alert (visual and sound) if the percentage exceeds 75%.

This approach is robust to depth map noise and works well for real-time close object alerting in driving assistance scenarios.
## 🟣 Depth-Based Alert Methodology

This project uses a MiDaS deep learning model to estimate a per-pixel depth map for each video frame. The depth value for each object is relative to the depth map as a whole (i.e., the estimated distance from the camera for each pixel), not relative to a specific reference point. The workflow is:

1. **Object Detection:** YOLO detects cars and pedestrians in the RGB frame, drawing bounding boxes.
2. **Depth Mapping:** The MiDaS model generates a depth map (single-channel, same size as the RGB frame) for the same frame.
3. **Object-Depth Association:** For each detected object, the code extracts the region of the depth map inside its bounding box.
4. **Alert Logic:**
  - The minimum depth value within each object's bounding box is computed.
  - If this minimum depth is below a configurable threshold, an alert is triggered ("Collision Warning!" or "Pedestrian Alert!").
  - The alert is only raised if a detected object is close, not just any close region in the depth map.
5. **Visualization:**
  - Bounding boxes and labels are drawn on both the RGB frame and the depth map, with alerting objects highlighted in red.
  - A colorbar shows the mapping from color to depth value.

**Note:** The depth map is only used for objects detected by YOLO. The alert system does not trigger for close regions unless they correspond to a detected car or person.






MiDaS outputs a raw depth map.
The code normalizes this map to 0–255.
Then, cv2.COLORMAP_INFERNO is applied, producing the colorized depth image.
allowed_bgr = inferno_lut[0:192] selects the BGR colors for the "closer" (left) part of this color

--=> INFERNO THRESHOLD 75% AND ALSO % OF PIXELS IN THAT BOX SATISFY COLOUR BHEFORE THRESHOLD OF INFERNO CHECKED











ALTERNATVE BETHDOLOGY BELOW WTH SSUES AND 2D ETC ABOVE DEPTH BETTER 3D

<h1 align="center">🚗 Advanced Driving Assistance</h1>

<p align="center">
<b>YOLO-based real-time collision and pedestrian alert system with lane detection overlays.</b>
</p>

---



## Alert Logic (Combined Color and Depth)

- The code uses MiDaS for monocular depth estimation and YOLO for object detection.
- The depth map is normalized and colorized using the Inferno colormap for visualization.
- **For each detected object, the code checks BOTH:**
  1. The mean of the raw depth values inside the object's bounding box (mean_depth).
  2. The percentage of bounding box pixels that match the allowed color range (from the colorized, normalized depth map).
- **An alert is triggered only if:**
  - mean_depth is less than the threshold (default: 0.4, lower means closer; see below for tuning)
  - AND at least 85% of the pixels match the color range (leftmost 75% of Inferno colormap)

This combined approach reduces false positives from per-frame normalization and ensures only truly close objects trigger alerts.

### Tuning the mean_depth threshold
- Lower threshold (e.g., 0.2): Only very close objects will trigger alerts (less sensitive).
- Higher threshold (e.g., 1.0, 2.0): Alerts will trigger for objects farther away (more sensitive).
- You may need to print mean_depth values for your camera/model to find a good threshold for your use case.

## Tuning

- Adjust the color match percentage (default: 85%) for stricter or looser color-based alerting.
- Adjust the mean depth threshold (default: 0.4) for stricter or looser real-world closeness.

## Known Limitations

- The color-based check is still relative to each frame, but the mean depth check ensures real-world closeness is required for an alert.

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

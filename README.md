# Pixel Estimation Archive (Archived Branch)

This branch is an archive for the pixel estimation approach to advanced driving assistance (collision avoidance). Lane detection code is also included but not tweaked.

## Methodology (Pixel Estimation)

1. **Video Input:** Load and process each frame from a driving video.
2. **Lane Detection:**
   - Convert frames to grayscale, blur, and detect edges.
   - Use perspective transform and histogram analysis to find lane lines.
   - Overlay detected lanes on the video.
3. **Object Detection:**
   - Use a YOLO model to detect cars and pedestrians.
   - Filter detections to cars and people only.
4. **Pixel Distance Estimation:**
   - Calculate the vertical pixel distance from detected objects to a fixed reference point (bottom center of the frame).
   - Highlight objects close to the reference point (potential collision or pedestrian alert).
5. **Alerts and Output:**
   - Show visual warnings and play a sound if a collision or pedestrian is detected within a threshold distance.
   - Draw bounding boxes, labels, and distance values on the video.
   - Save the processed video with overlays and alerts.

---

---
**Note:** This approach was archived because 2D pixel distances do not represent real-world distance and are sensitive to camera position, zoom, and perspective. Depth-based methods provide more accurate, robust, and real-world relevant distance estimation for driving assistance.

This branch is for reference only. No further development will occur here.

## Notes & Limitations

- Pixel-based distances are **not** real-world meters; always recalibrate if camera/lens/zoom changes.
- For best accuracy, keep the camera fixed and avoid digital zoom.
- Lane detection uses OpenCV pipeline (see `edge_detection.py`).
- The reference is a fixed dot on the screen (bottom-center, usually at the bonnet edge). Pixel distances are measured from this fixed dot to the **closest edge** of each detected object's bounding box (not the centroid).
- If you click on a detected object in the display window, the script will show the vertical pixel distance from that object to all others (for visualization only, not for alerts).
- When a collision or pedestrian alert is triggered, an alert sound (`alert_sound.mp3`) will play (if present in the script directory). The sound will not overlap if multiple alerts occur in quick succession.
- In 2D pixel estimation, the reference point for distance measurement can be any fixed location on the screen (such as the bottom-center). No object detection is needed for this reference point, and OpenCV is not required for its placement—pixel distances are simply measured on-screen between this point and detected objects.
---

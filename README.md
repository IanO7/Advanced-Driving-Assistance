**Important Calibration Note:**
- If you change the camera zoom or field of view, you must recalibrate the pixel thresholds for collision and pedestrian alerts. The same pixel values do not represent the same real-world distance if the zoom changes, even if the video resolution stays the same.**CHECK FOR SAME VIDEO SIZE DOES IT WORK, SO FILTER AS 1920X1080 FIRST THEN OK, 
BUT NEED TO CONSIDER ZOOM SINCE(If you zoom in, objects appear larger and closer in the frame, so the same pixel distance covers less real-world space.
If you zoom out, objects appear smaller and farther, so the same pixel distance covers more real-world space.)

  =>because assume 5m distance, then zoom in looks closer, but xoom out is further, but actual fact is safe distance
  => HOW TO ENSURE SCALE AND ZOOM UNAFFECTED ACROSS VIDEOS??
  => Scale videos also to 1920 x 1080

  => Tell users cant zoom in, then different lens will still have different params (but 1 var less), so adjust dynamic??
  => TECHNICALLY CHECK AGAIN IF POINT DOT TO IMAGE IS CRASH CAN STILL ESTIMATE CLOSENES??
  => Cloest edge is best then will mitage zoomed in or out since assuming closest if zoomed case and if zoomed out then still ok?
  => EUCLIDEAN DISTANCE BETTER SINCE GENERALISED AND FOR DIAGONAL CARS ALSO TO ALERT COLLISION, BUT ZOOMED ISSUE STILL NEED TO FIX, bottom edge makes sense also since cloest to bumter to bumper


  *how to ensure bothvideos pedestrian & car crash generalised and close enough?, check is same pixels when close enough??





# Adavanced Driving Assistance

This repository contains:
- `pixel_estimation.py` — main script: runs YOLO detection, draws lane overlay (Automatic Addison pipeline), and raises pedestrian/collision alerts based on pixel distances.
- `edge_detection.py` — helper functions for thresholding, Sobel/magnitude, Gaussian blur used by lane detection.



**BEST TO FINALISE IS STATE AND VERFY FROM ULTRALYTICS OPEN CV THEN PIXEL BASED STEPS, THEN MY STEP IS THAT BUT SIMPLER TO MEASURE DISTANCE SAME DELA BUT TO REF OBJECT EAIER INSTEAD MODEIFIES!

Quick summary**UPDATE ENSURE FINALSE SYNC AVEC CODE
- Ultralytics YOLO11, as used here, measures distances in pixels between detected object boxes within the video frame. Objects (cars, people) are identified using YOLO and OpenCV. Then, the original code measures pixel distances between two selected objects.
- In this implementation, instead of always measuring between two detected objects (via OpenCV), a fixed reference point (a specific pixel location on the screen, e.g., the red dot at 80% frame height) is used. Pixel distances are then measured from this reference point to the centroids of all detected cars and people, following the same pixel-based logic as the original Ultralytics approach. i.e., Because the original implementation measures pixel distances on the screen, the reference point can be any fixed pixel location—no object detection is required to define it, then all other objects follow original version to that point for pixel calculation
- Assuming that edge of bonnet is the bottom reference point, ignores everything below that point as bonnet, and abve is edge to other objects iwthin THRESHOLD **seems 450 best for car crash vid BUT PEDESTRIAN VID DIFFERENT SCALES?, heck pedestrian ok!?


**OPEN CV THRESHLDS THEN PIXEL THRESHOLDS CLEARLY STATE (455 OK??) => ACCURACCIES (BASED ON CRASH VIDS OR REAL DISTANCES APPROXIMATIONS??)



**BELOW IS IDEAL, CHECK CODE UPDATE PERFECT SYNC FINALISE
Color Coding for Car Detection
- **Red**: Any car within the collision threshold distance from the bottom-center reference point (red dot).
- **Green**: All other detected cars (not in collision range to the bottom-center reference point).

Pedestrian logic:
- **Magenta**: Pedestrian alert—only for pedestrians close to the bottom-center reference point (within a set pixel threshold).
- **Cyan**: All other detected pedestrians.


- Lane detection**UPDATE

Requirements
- Python 3.8+ (confirm with your environment)
- Packages (install into your conda/venv):
  - opencv-python
  - numpy
  - ultralytics

How to run
1. **ENV & DEPENDEICES 


- Adjust alert thresholds:
  - Collision threshold is in pixels and defined in `pixel_estimation.py` as `COLLISION_THRESHOLD`. Larger values mean more alerts (objects farther away trigger alerts); smaller values mean fewer alerts.
  - Pedestrian threshold is `PEDESTRIAN_THRESHOLD` (also in pixels).
  - Important: the code measures distances to the bottom-center of the frame. If your camera view includes the car bonnet at the bottom, distances near the bottom represent extremely close-range; you may need to increase the threshold to still trigger alerts. If bonnet is not visible (road to bottom edge), the same threshold will behave differently (trigger earlier).



Contact / License
- This README was auto-generated. Adapt and improve as you tune the project.
- No license included.

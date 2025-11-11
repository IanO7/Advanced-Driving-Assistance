from ldw import ldw_overlay
import threading
import os
try:
    from playsound import playsound
except ImportError:
    def playsound(path):
        pass  # fallback: do nothing if playsound is not installed

# Prevent overlapping alert sounds (copied from pixel_estimation.py)
alert_sound_lock = threading.Lock()
alert_sound_playing = False

def play_alert_once():
    global alert_sound_playing
    with alert_sound_lock:
        if alert_sound_playing:
            return
        alert_sound_playing = True
    try:
        playsound(os.path.join(os.path.dirname(__file__), 'assets', 'alert_sound.mp3'))
    except Exception:
        pass
    finally:
        with alert_sound_lock:
            alert_sound_playing = False
import cv2
import torch
import numpy as np
import warnings
from ultralytics import YOLO
import os
warnings.filterwarnings("ignore", category=FutureWarning)


class MiDaS:
    @staticmethod
    def show_birds_eye_view(frame, car_boxes, window_name="BirdsEyeView", yolo_results=None):
        # --- Bird's Eye View Icon Logic (matches README) ---
        # For each detected object, the system follows a two-stage logic:
        #   1. Stage 1: Awareness (Green Icon)
        #      - If an object (car, bus, truck, or pedestrian) is detected by YOLO/OpenCV,
        #        a green icon is shown in the bird’s eye view for that zone.
        #      - This indicates presence only—no distance or depth check is performed, and no alert sound or warning is triggered.
        #   2. Stage 2: Collision Risk (Red Icon)
        #      - The system examines the corresponding region in the depth map for each detected object.
        #      - If more than 75% of the pixels in the object's bounding box match the close-range Inferno colormap,
        #        a red icon is shown (alert), and a visual and audio warning is triggered.
        #      - The red icon always takes priority over green in a given zone.

        # Parameters for the bird's eye view
        view_w, view_h = 300, 400
        car_w, car_h = 60, 100
        zone_h = 120
        margin = 20
        # Create blank white image
        view = np.ones((view_h, view_w, 3), dtype=np.uint8) * 255
        # Draw car image in the center bottom
        car_x = view_w // 2 - car_w // 2
        car_y = view_h - car_h - margin
        car_img_path = os.path.join(os.path.dirname(__file__), 'assets', 'birds_eye_view_car.png')
        car_img_path_detected = os.path.join(os.path.dirname(__file__), 'assets', 'red_car.png')
        if os.path.exists(car_img_path):
            car_img = cv2.imread(car_img_path, cv2.IMREAD_UNCHANGED)
            if car_img is not None:
                car_img = cv2.resize(car_img, (car_w, car_h))
                # If PNG with alpha, blend
                if car_img.shape[2] == 4:
                    alpha = car_img[:,:,3] / 255.0
                    for c in range(3):
                        view[car_y:car_y+car_h, car_x:car_x+car_w, c] = (
                            alpha * car_img[:,:,c] + (1-alpha) * view[car_y:car_y+car_h, car_x:car_x+car_w, c]
                        ).astype(np.uint8)
                else:
                    view[car_y:car_y+car_h, car_x:car_x+car_w] = car_img[:,:,:3]
        else:
            cv2.rectangle(view, (car_x, car_y), (car_x + car_w, car_y + car_h), (50, 50, 50), -1)
            cv2.putText(view, "YOU", (car_x + 5, car_y + car_h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        # Define 3 zones in front of the car
        zone_top = car_y - zone_h
        zones = [
            ((margin, zone_top), (car_x, car_y)),  # Left
            ((car_x, zone_top), (car_x + car_w, car_y)),  # Center
            ((car_x + car_w, zone_top), (view_w - margin, car_y)),  # Right
        ]
        zone_colors = [(200,200,200)]*3
        # Map car_boxes to zones and track which type of alert (car/bus/truck or pedestrian)
        zone_alert_types = [None, None, None]  # 'car', 'bus', 'truck', 'pedestrian', or None
        zone_green_types = [None, None, None]  # For awareness-only (green) icons
        # First, fill in red alert zones as before
        # (Red is checked/set first, so it always takes priority over green)
        for box in car_boxes:
            x1, y1, x2, y2, alert_type = box if len(box) == 5 else (*box, 'car')
            box_cx = (x1 + x2) / 2
            w = frame.shape[1]
            if box_cx < w/3:
                zone_colors[0] = (0,0,255)
                zone_alert_types[0] = alert_type
            elif box_cx < 2*w/3:
                zone_colors[1] = (0,0,255)
                zone_alert_types[1] = alert_type
            else:
                zone_colors[2] = (0,0,255)
                zone_alert_types[2] = alert_type
        # Now, if yolo_results is provided, fill in green awareness icons for any detected object (no alert)
        # (Green is only set if there is no red alert in that zone)
        if yolo_results is not None and hasattr(yolo_results, 'boxes') and yolo_results.boxes is not None:
            all_class_ids = yolo_results.boxes.cls.cpu().numpy().astype(int)
            boxes_all = yolo_results.boxes.xyxy.cpu().numpy()
            for box, cls_id in zip(boxes_all, all_class_ids):
                if cls_id not in [0, 2, 5, 7]:
                    continue
                x1, y1, x2, y2 = map(int, box)
                box_cx = (x1 + x2) / 2
                w = frame.shape[1]
                if box_cx < w/3:
                    zone = 0
                elif box_cx < 2*w/3:
                    zone = 1
                else:
                    zone = 2
                # Only set green if not already red alert in that zone
                if zone_alert_types[zone] is None:
                    if cls_id == 0:
                        zone_green_types[zone] = 'pedestrian'
                    elif cls_id == 2:
                        zone_green_types[zone] = 'car'
                    elif cls_id == 5:
                        zone_green_types[zone] = 'bus'
                    elif cls_id == 7:
                        zone_green_types[zone] = 'truck'
        red_car_img_path = os.path.join(os.path.dirname(__file__), 'assets', 'red_car.png')
        red_person_img_path = os.path.join(os.path.dirname(__file__), 'assets', 'red_person.png')
        red_bus_img_path = os.path.join(os.path.dirname(__file__), 'assets', 'red_bus.png')
        green_car_img_path = os.path.join(os.path.dirname(__file__), 'assets', 'green_car.png')
        green_person_img_path = os.path.join(os.path.dirname(__file__), 'assets', 'green_person.png')
        green_bus_img_path = os.path.join(os.path.dirname(__file__), 'assets', 'green_bus.png')
        red_car_img = None
        red_person_img = None
        red_bus_img = None
        green_car_img = None
        green_person_img = None
        green_bus_img = None
        if os.path.exists(red_car_img_path):
            red_car_img = cv2.imread(red_car_img_path, cv2.IMREAD_UNCHANGED)
        if os.path.exists(red_person_img_path):
            red_person_img = cv2.imread(red_person_img_path, cv2.IMREAD_UNCHANGED)
        if os.path.exists(red_bus_img_path):
            red_bus_img = cv2.imread(red_bus_img_path, cv2.IMREAD_UNCHANGED)
        if os.path.exists(green_car_img_path):
            green_car_img = cv2.imread(green_car_img_path, cv2.IMREAD_UNCHANGED)
        if os.path.exists(green_person_img_path):
            green_person_img = cv2.imread(green_person_img_path, cv2.IMREAD_UNCHANGED)
        if os.path.exists(green_bus_img_path):
            green_bus_img = cv2.imread(green_bus_img_path, cv2.IMREAD_UNCHANGED)
        for i, ((x0, y0), (x1, y1)) in enumerate(zones):
            # Draw zone background
            cv2.rectangle(view, (x0, y0), (x1, y1), (220,220,220), -1)
            cv2.rectangle(view, (x0, y0), (x1, y1), (100,100,100), 2)
            # Center position for car image in this zone
            zone_cx = (x0 + x1) // 2
            zone_cy = (y0 + y1) // 2 + 10
            y_start = zone_cy - car_h//2
            x_start = zone_cx - car_w//2
            if zone_colors[i] == (0,0,255):
                # Show red car, red bus, or red person image if available, based on alert type
                alert_type = zone_alert_types[i]
                img_to_use = None
                if alert_type == 'pedestrian' and red_person_img is not None:
                    img_to_use = red_person_img
                elif alert_type == 'car' and red_car_img is not None:
                    img_to_use = red_car_img
                elif alert_type == 'bus' and red_bus_img is not None:
                    img_to_use = red_bus_img
                elif alert_type == 'truck' and red_bus_img is not None:
                    img_to_use = red_bus_img
                if img_to_use is not None:
                    car_img_zone = cv2.resize(img_to_use, (car_w, car_h))
                    if car_img_zone.shape[2] == 4:
                        alpha = car_img_zone[:,:,3] / 255.0
                        for c in range(3):
                            view[y_start:y_start+car_h, x_start:x_start+car_w, c] = (
                                alpha * car_img_zone[:,:,c] + (1-alpha) * view[y_start:y_start+car_h, x_start:x_start+car_w, c]
                            ).astype(np.uint8)
                    else:
                        view[y_start:y_start+car_h, x_start:x_start+car_w] = car_img_zone[:,:,:3]
                    # Draw red border
                    cv2.rectangle(view, (x_start, y_start), (x_start+car_w, y_start+car_h), (0,0,255), 4)
                elif os.path.exists(car_img_path) and car_img is not None:
                    car_img_zone = cv2.resize(car_img, (car_w, car_h))
                    if car_img_zone.shape[2] == 4:
                        alpha = car_img_zone[:,:,3] / 255.0
                        for c in range(3):
                            view[y_start:y_start+car_h, x_start:x_start+car_w, c] = (
                                alpha * car_img_zone[:,:,c] + (1-alpha) * view[y_start:y_start+car_h, x_start:x_start+car_w, c]
                            ).astype(np.uint8)
                    else:
                        view[y_start:y_start+car_h, x_start:x_start+car_w] = car_img_zone[:,:,:3]
                    cv2.rectangle(view, (x_start, y_start), (x_start+car_w, y_start+car_h), (0,0,255), 4)
                else:
                    # Fallback: just draw red rectangle
                    cv2.rectangle(view, (x0, y0), (x1, y1), (0,0,255), 4)
            elif zone_green_types[i] is not None:
                # Show green awareness icon (no alert, just detected by YOLO)
                green_type = zone_green_types[i]
                img_to_use = None
                if green_type == 'pedestrian' and green_person_img is not None:
                    img_to_use = green_person_img
                elif green_type == 'car' and green_car_img is not None:
                    img_to_use = green_car_img
                elif green_type == 'bus' and green_bus_img is not None:
                    img_to_use = green_bus_img
                elif green_type == 'truck' and green_bus_img is not None:
                    img_to_use = green_bus_img
                if img_to_use is not None:
                    car_img_zone = cv2.resize(img_to_use, (car_w, car_h))
                    if car_img_zone.shape[2] == 4:
                        alpha = car_img_zone[:,:,3] / 255.0
                        for c in range(3):
                            view[y_start:y_start+car_h, x_start:x_start+car_w, c] = (
                                alpha * car_img_zone[:,:,c] + (1-alpha) * view[y_start:y_start+car_h, x_start:x_start+car_w, c]
                            ).astype(np.uint8)
                    else:
                        view[y_start:y_start+car_h, x_start:x_start+car_w] = car_img_zone[:,:,:3]
        # Draw car image again at the bottom (your car)
        if os.path.exists(car_img_path) and 'car_img' in locals() and car_img is not None:
            if car_img.shape[2] == 4:
                alpha = car_img[:,:,3] / 255.0
                for c in range(3):
                    view[car_y:car_y+car_h, car_x:car_x+car_w, c] = (
                        alpha * car_img[:,:,c] + (1-alpha) * view[car_y:car_y+car_h, car_x:car_x+car_w, c]
                    ).astype(np.uint8)
            else:
                view[car_y:car_y+car_h, car_x:car_x+car_w] = car_img[:,:,:3]
        else:
            cv2.rectangle(view, (car_x, car_y), (car_x + car_w, car_y + car_h), (50, 50, 50), -1)
        cv2.imshow(window_name, view)
        # Parameters for the bird's eye view
        view_w, view_h = 300, 400
        car_w, car_h = 60, 100
        zone_h = 120
        margin = 20
        # Create blank white image
        view = np.ones((view_h, view_w, 3), dtype=np.uint8) * 255
        # Draw car image in the center bottom
        car_x = view_w // 2 - car_w // 2
        car_y = view_h - car_h - margin
        car_img_path = os.path.join(os.path.dirname(__file__), 'assets', 'birds_eye_view_car.png')
        car_img_path_detected = os.path.join(os.path.dirname(__file__), 'assets', 'red_car.png')
        if os.path.exists(car_img_path):
            car_img = cv2.imread(car_img_path, cv2.IMREAD_UNCHANGED)
            if car_img is not None:
                car_img = cv2.resize(car_img, (car_w, car_h))
                # If PNG with alpha, blend
                if car_img.shape[2] == 4:
                    alpha = car_img[:,:,3] / 255.0
                    for c in range(3):
                        view[car_y:car_y+car_h, car_x:car_x+car_w, c] = (
                            alpha * car_img[:,:,c] + (1-alpha) * view[car_y:car_y+car_h, car_x:car_x+car_w, c]
                        ).astype(np.uint8)
                else:
                    view[car_y:car_y+car_h, car_x:car_x+car_w] = car_img[:,:,:3]
        # else fallback: draw rectangle
        else:
            cv2.rectangle(view, (car_x, car_y), (car_x + car_w, car_y + car_h), (50, 50, 50), -1)
            cv2.putText(view, "YOU", (car_x + 5, car_y + car_h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        # Define 3 zones in front of the car
        zone_top = car_y - zone_h
        zones = [
            ((margin, zone_top), (car_x, car_y)),  # Left
            ((car_x, zone_top), (car_x + car_w, car_y)),  # Center
            ((car_x + car_w, zone_top), (view_w - margin, car_y)),  # Right
        ]
        zone_colors = [(200,200,200)]*3
        # Map car_boxes to zones and track which type of alert (car/bus/truck or pedestrian)
        zone_alert_types = [None, None, None]  # 'car', 'bus', 'truck', 'pedestrian', or None
        zone_green_types = [None, None, None]  # For awareness-only (green) icons
        # First, fill in red alert zones as before
        for box in car_boxes:
            x1, y1, x2, y2, alert_type = box if len(box) == 5 else (*box, 'car')
            box_cx = (x1 + x2) / 2
            w = frame.shape[1]
            if box_cx < w/3:
                zone_colors[0] = (0,0,255)
                zone_alert_types[0] = alert_type
            elif box_cx < 2*w/3:
                zone_colors[1] = (0,0,255)
                zone_alert_types[1] = alert_type
            else:
                zone_colors[2] = (0,0,255)
                zone_alert_types[2] = alert_type
        # Now, if yolo_results is provided, fill in green awareness icons for any detected object (no alert)
        if yolo_results is not None and hasattr(yolo_results, 'boxes') and yolo_results.boxes is not None:
            all_class_ids = yolo_results.boxes.cls.cpu().numpy().astype(int)
            boxes_all = yolo_results.boxes.xyxy.cpu().numpy()
            for box, cls_id in zip(boxes_all, all_class_ids):
                if cls_id not in [0, 2, 5, 7]:
                    continue
                x1, y1, x2, y2 = map(int, box)
                box_cx = (x1 + x2) / 2
                w = frame.shape[1]
                if box_cx < w/3:
                    zone = 0
                elif box_cx < 2*w/3:
                    zone = 1
                else:
                    zone = 2
                # Only set green if not already red alert in that zone
                if zone_alert_types[zone] is None:
                    if cls_id == 0:
                        zone_green_types[zone] = 'pedestrian'
                    elif cls_id == 2:
                        zone_green_types[zone] = 'car'
                    elif cls_id == 5:
                        zone_green_types[zone] = 'bus'
                    elif cls_id == 7:
                        zone_green_types[zone] = 'truck'
        # Draw car image in each zone (if car detected, highlight with red border)

        red_car_img_path = os.path.join(os.path.dirname(__file__), 'assets', 'red_car.png')
        red_person_img_path = os.path.join(os.path.dirname(__file__), 'assets', 'red_person.png')
        red_bus_img_path = os.path.join(os.path.dirname(__file__), 'assets', 'red_bus.png')
        green_car_img_path = os.path.join(os.path.dirname(__file__), 'assets', 'green_car.png')
        green_person_img_path = os.path.join(os.path.dirname(__file__), 'assets', 'green_person.png')
        green_bus_img_path = os.path.join(os.path.dirname(__file__), 'assets', 'green_bus.png')
        red_car_img = None
        red_person_img = None
        red_bus_img = None
        green_car_img = None
        green_person_img = None
        green_bus_img = None
        if os.path.exists(red_car_img_path):
            red_car_img = cv2.imread(red_car_img_path, cv2.IMREAD_UNCHANGED)
        if os.path.exists(red_person_img_path):
            red_person_img = cv2.imread(red_person_img_path, cv2.IMREAD_UNCHANGED)
        if os.path.exists(red_bus_img_path):
            red_bus_img = cv2.imread(red_bus_img_path, cv2.IMREAD_UNCHANGED)
        if os.path.exists(green_car_img_path):
            green_car_img = cv2.imread(green_car_img_path, cv2.IMREAD_UNCHANGED)
        if os.path.exists(green_person_img_path):
            green_person_img = cv2.imread(green_person_img_path, cv2.IMREAD_UNCHANGED)
        if os.path.exists(green_bus_img_path):
            green_bus_img = cv2.imread(green_bus_img_path, cv2.IMREAD_UNCHANGED)

        for i, ((x0, y0), (x1, y1)) in enumerate(zones):
            # Draw zone background
            cv2.rectangle(view, (x0, y0), (x1, y1), (220,220,220), -1)
            cv2.rectangle(view, (x0, y0), (x1, y1), (100,100,100), 2)
            # Center position for car image in this zone
            zone_cx = (x0 + x1) // 2
            zone_cy = (y0 + y1) // 2 + 10
            y_start = zone_cy - car_h//2
            x_start = zone_cx - car_w//2
            if zone_colors[i] == (0,0,255):
                # Show red car, red bus, or red person image if available, based on alert type
                alert_type = zone_alert_types[i]
                img_to_use = None
                if alert_type == 'pedestrian' and red_person_img is not None:
                    img_to_use = red_person_img
                elif alert_type == 'car' and red_car_img is not None:
                    img_to_use = red_car_img
                elif alert_type == 'bus' and red_bus_img is not None:
                    img_to_use = red_bus_img
                elif alert_type == 'truck' and red_bus_img is not None:
                    img_to_use = red_bus_img
                if img_to_use is not None:
                    car_img_zone = cv2.resize(img_to_use, (car_w, car_h))
                    if car_img_zone.shape[2] == 4:
                        alpha = car_img_zone[:,:,3] / 255.0
                        for c in range(3):
                            view[y_start:y_start+car_h, x_start:x_start+car_w, c] = (
                                alpha * car_img_zone[:,:,c] + (1-alpha) * view[y_start:y_start+car_h, x_start:x_start+car_w, c]
                            ).astype(np.uint8)
                    else:
                        view[y_start:y_start+car_h, x_start:x_start+car_w] = car_img_zone[:,:,:3]
                    # Draw red border
                    cv2.rectangle(view, (x_start, y_start), (x_start+car_w, y_start+car_h), (0,0,255), 4)
                elif os.path.exists(car_img_path) and car_img is not None:
                    car_img_zone = cv2.resize(car_img, (car_w, car_h))
                    if car_img_zone.shape[2] == 4:
                        alpha = car_img_zone[:,:,3] / 255.0
                        for c in range(3):
                            view[y_start:y_start+car_h, x_start:x_start+car_w, c] = (
                                alpha * car_img_zone[:,:,c] + (1-alpha) * view[y_start:y_start+car_h, x_start:x_start+car_w, c]
                            ).astype(np.uint8)
                    else:
                        view[y_start:y_start+car_h, x_start:x_start+car_w] = car_img_zone[:,:,:3]
                    cv2.rectangle(view, (x_start, y_start), (x_start+car_w, y_start+car_h), (0,0,255), 4)
                else:
                    # Fallback: just draw red rectangle
                    cv2.rectangle(view, (x0, y0), (x1, y1), (0,0,255), 4)
            elif zone_green_types[i] is not None:
                # Show green awareness icon (no alert, just detected by YOLO)
                green_type = zone_green_types[i]
                img_to_use = None
                if green_type == 'pedestrian' and green_person_img is not None:
                    img_to_use = green_person_img
                elif green_type == 'car' and green_car_img is not None:
                    img_to_use = green_car_img
                elif green_type == 'bus' and green_bus_img is not None:
                    img_to_use = green_bus_img
                elif green_type == 'truck' and green_bus_img is not None:
                    img_to_use = green_bus_img
                if img_to_use is not None:
                    car_img_zone = cv2.resize(img_to_use, (car_w, car_h))
                    if car_img_zone.shape[2] == 4:
                        alpha = car_img_zone[:,:,3] / 255.0
                        for c in range(3):
                            view[y_start:y_start+car_h, x_start:x_start+car_w, c] = (
                                alpha * car_img_zone[:,:,c] + (1-alpha) * view[y_start:y_start+car_h, x_start:x_start+car_w, c]
                            ).astype(np.uint8)
                    else:
                        view[y_start:y_start+car_h, x_start:x_start+car_w] = car_img_zone[:,:,:3]
        # Draw car image again at the bottom (your car)
        if os.path.exists(car_img_path) and car_img is not None:
            if car_img.shape[2] == 4:
                alpha = car_img[:,:,3] / 255.0
                for c in range(3):
                    view[car_y:car_y+car_h, car_x:car_x+car_w, c] = (
                        alpha * car_img[:,:,c] + (1-alpha) * view[car_y:car_y+car_h, car_x:car_x+car_w, c]
                    ).astype(np.uint8)
            else:
                view[car_y:car_y+car_h, car_x:car_x+car_w] = car_img[:,:,:3]
        else:
            cv2.rectangle(view, (car_x, car_y), (car_x + car_w, car_y + car_h), (50, 50, 50), -1)
    """Performs monocular depth estimation using Intel Labs MiDaS models.

    This class provides utilities to load a pre-trained MiDaS model,
    apply image transforms, generate depth maps, and normalize the results
    for visualization. It also supports real-time depth inference from video streams.

    Attributes:
        midas (torch.nn.Module): The MiDaS model instance.
        transform (callable): The preprocessing transform for input images.
        model_type (str): The type of MiDaS model to load.
        device (torch.device): The computation device (CPU or CUDA).
    """

    def __init__(self, model_type: str):
        """Initializes the MiDaS depth estimation class.

        Args:
            model_type (str): The model variant to use.
                Supported values:
                - "DPT_Large": Highest accuracy, slowest speed.
                - "DPT_Hybrid": Balanced accuracy and speed.
                - "MiDaS_small": Fastest, lowest accuracy.
        """
        self.midas = None
        self.transform = None
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()

    def load_model(self) -> None:
        """Loads the MiDaS model from the PyTorch Hub."""
        self.midas = torch.hub.load("intel-isl/MiDaS", self.model_type)
        self.midas.to(self.device).eval()

    def transforms(self):
        """Retrieves the appropriate image preprocessing transform."""
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if self.model_type in ["DPT_Large", "DPT_Hybrid"]:
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform
        return self.transform

    def depth_map(self, batch: torch.Tensor, img: np.ndarray) -> np.ndarray:
        """Generates a depth map for a given input image batch."""
        with torch.no_grad():
            prediction = self.midas(batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        return prediction.cpu().numpy()

    @staticmethod
    def normalize_depth(depth_map: np.ndarray):
        """Normalizes and colorizes a depth map for visualization. Returns colorized map and min/max values."""
        depth_min, depth_max = depth_map.min(), depth_map.max()
        normalized = (depth_map - depth_min) / (depth_max - depth_min)
        normalized = (normalized * 255).astype(np.uint8)
        colorized = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
        return colorized, depth_min, depth_max

    @staticmethod
    def draw_depth_colorbar(image, vmin, vmax, colormap=cv2.COLORMAP_INFERNO, width=30, height=200, margin=10, decimals=2):
        """Draws a vertical colorbar on the right side of the image showing the depth range."""
        bar = np.linspace(0, 255, height).astype(np.uint8)[::-1].reshape(-1, 1)
        bar_img = np.repeat(bar, width, axis=1)
        bar_color = cv2.applyColorMap(bar_img, colormap)
        # Add fixed 0-255 labels for the colormap
        bar_color = bar_color.copy()
        # Place 0 at bottom, 128 at middle, 255 at top
        label_positions = [(height-1, 0), ((height-1)//2, 128), (0, 255)]
        for y, val in label_positions:
            label = f"{val}"
            cv2.putText(bar_color, label, (width+5, y+8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        # Place bar on right side of image
        h, w = image.shape[:2]
        overlay = image.copy()
        y0 = margin
        y1 = y0 + height
        x0 = w - width - 60
        x1 = x0 + width
        if y1 > h:
            y0 = h - height - margin
            y1 = h - margin
        overlay[y0:y1, x0:x1] = bar_color
        return overlay

    @staticmethod
    def stack_frames(frame1: np.ndarray, frame2: np.ndarray, width: int = 1280, height: int = 640) -> np.ndarray:
        """Safely stacks two frames horizontally with resizing to a fixed display resolution."""
        h, w = frame1.shape[:2]
        frame2 = cv2.resize(frame2, (w, h))
        combined = np.hstack((frame1, frame2))
        combined = cv2.resize(combined, (width, height))
        return combined

    def infer_video(self, source: str = 0, output_path: str = None, display: bool = True) -> None:
        """Performs real-time depth estimation on a video stream."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")

        width = 1280
        height = 640
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        transform = self.transforms()

        # Load YOLO model for object detection
        yolo_model = YOLO("yolo11n.pt")
        class_names = yolo_model.model.names if hasattr(yolo_model.model, 'names') else [str(i) for i in range(100)]

        print("🚀 Starting video depth inference... Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_batch = transform(rgb).to(self.device)

            depth = self.depth_map(input_batch, rgb)
            colored_depth, dmin, dmax = self.normalize_depth(depth)
            # We'll draw boxes on a copy so the colorbar is always on top
            colored_depth_with_boxes = colored_depth.copy()


            # --- Object Detection and Alert Logic ---
            results = yolo_model(frame)[0]
            if results.boxes is not None:
                all_class_ids = results.boxes.cls.cpu().numpy().astype(int)
                boxes_all = results.boxes.xyxy.cpu().numpy()
            else:
                all_class_ids = []
                boxes_all = np.empty((0, 4))

            alert_triggered = False
            alert_texts = set()
            alert_indices = set()
            # Precompute allowed BGR set for alerting (Inferno colormap, indices 58-255)
            allowed_bgr_set = set(tuple(map(int, np.array(bgr).flatten())) for bgr in cv2.applyColorMap(np.arange(0, 256, dtype=np.uint8), cv2.COLORMAP_INFERNO)[58:256])

            car_boxes_birdseye = []
            # Exclude objects completely below the yellow guide line (bonnet area)
            guide_y = int(frame.shape[0] * 0.95)
            for idx, (box, cls_id) in enumerate(zip(boxes_all, all_class_ids)):
                if cls_id not in [0, 2, 5, 7]:
                    continue
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1]-1, x2), min(frame.shape[0]-1, y2)
                # Exclude if the entire box is below the yellow line
                if y1 > guide_y:
                    continue
                color_region = colored_depth[y1:y2, x1:x2]
                if color_region.size == 0:
                    continue
                color_region_flat = color_region.reshape(-1, 3)
                match_count = sum(tuple(pixel) in allowed_bgr_set for pixel in color_region_flat)
                total_pixels = color_region_flat.shape[0]
                color_match_ratio = match_count / total_pixels if total_pixels > 0 else 0
                is_alert = color_match_ratio > 0.75
                print(f"Object: {class_names[cls_id] if cls_id < len(class_names) else str(cls_id)}, match: {match_count}, total: {total_pixels}, color_match_ratio: {color_match_ratio:.2f}, alert: {is_alert}")
                if is_alert:
                    alert_triggered = True
                    alert_indices.add(idx)
                    if cls_id == 2:
                        alert_texts.add("Collision Warning! (Car)")
                        car_boxes_birdseye.append((x1, y1, x2, y2, 'car'))
                    elif cls_id == 0:
                        alert_texts.add("Pedestrian Alert!")
                        car_boxes_birdseye.append((x1, y1, x2, y2, 'pedestrian'))
                    elif cls_id == 5:
                        alert_texts.add("Bus Alert!")
                        car_boxes_birdseye.append((x1, y1, x2, y2, 'bus'))
                    elif cls_id == 7:
                        alert_texts.add("Truck Alert!")
                        car_boxes_birdseye.append((x1, y1, x2, y2, 'truck'))

            # Draw all boxes, highlight alert ones in red
            for idx, (box, cls_id) in enumerate(zip(boxes_all, all_class_ids)):
                if cls_id not in [0, 2, 5, 7]:
                    continue
                # Use same color logic for all alert types
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1]-1, x2), min(frame.shape[0]-1, y2)
                # Exclude if the entire box is below the yellow line
                if y1 > guide_y:
                    continue
                depth_region = depth[y1:y2, x1:x2]
                if depth_region.size > 0:
                    mean_depth = float(np.mean(depth_region))
                else:
                    mean_depth = float('nan')
                if idx in alert_indices:
                    color = (0, 0, 255)  # Red for alert (car or pedestrian)
                else:
                    color = (0, 255, 0) if cls_id == 2 else (255, 255, 0)
                label = f"{class_names[cls_id] if cls_id < len(class_names) else str(cls_id)}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.rectangle(colored_depth_with_boxes, (x1, y1), (x2, y2), color, 2)
                cv2.putText(colored_depth_with_boxes, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if alert_triggered:
                threading.Thread(target=play_alert_once, daemon=True).start()
                for i, text in enumerate(sorted(alert_texts)):
                    cv2.putText(frame, text, (30, 60 + i * 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

            # Add colorbar after drawing boxes so it is always on top
            colored_depth_with_boxes = self.draw_depth_colorbar(colored_depth_with_boxes, dmin, dmax)


            # Lane Departure Warning overlay (left side)
            ldw_frame = ldw_overlay(frame)
            # Draw the guide line and text on the LDW overlay only (not on the frame used for detection)
            guide_y = int(ldw_frame.shape[0] * 0.95)
            cv2.line(ldw_frame, (0, guide_y), (ldw_frame.shape[1], guide_y), (0, 255, 255), 2)
            cv2.putText(ldw_frame, 'Align bonnet with this line for best results (BELOW NOT CONSIDERED IN ALGORITHM)', (10, guide_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            # Resize LDW overlay and depth map to same size
            ldw_frame_resized = cv2.resize(ldw_frame, (width // 2, height))
            depth_resized = cv2.resize(colored_depth_with_boxes, (width // 2, height))
            # Concatenate LDW overlay (left) and depth map (right)
            combined = np.hstack((ldw_frame_resized, depth_resized))

            if display:
                cv2.imshow("MiDaS Depth Estimation (Press 'q' to exit)", combined)
                # Show bird's eye view window (pass yolo results for awareness icons)
                self.show_birds_eye_view(frame, car_boxes_birdseye, yolo_results=results)

            if writer:
                writer.write(combined)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("✅ Inference completed and resources released.")



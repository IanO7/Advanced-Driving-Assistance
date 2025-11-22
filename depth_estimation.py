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
import time
from ultralytics import YOLO
warnings.filterwarnings("ignore", category=FutureWarning)

# Default sensitivity cutoff for bird's-eye alerting (Inferno colormap index)
# Lower values => more aggressive (objects further away may trigger). 58 is the
# project-recommended default determined experimentally.
BIRDSEYE_SENSITIVITY_DEFAULT = 58


class LatestFrameReader:
    """Continuously grabs frames in a background thread and exposes only the most recent.

    This avoids processing a backlog when inference is slower than incoming frames.
    If inference of one frame takes 1s, the next processed frame will be the *current*
    frame at that moment, not buffered older frames.
    Note: Intended for live streams/cameras. Video files are read sequentially
    in infer_video() to avoid EOF freeze and ensure deterministic playback.
    """
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")
        # Try to shrink internal buffer if backend honors it
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self.lock = threading.Lock()
        self.latest_frame = None
        self.running = True
        self.reader_thread = threading.Thread(target=self._update, daemon=True)
        self.reader_thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                # Source ended or error; stop loop
                self.running = False
                # Ensure get_latest() returns None so callers can exit cleanly
                with self.lock:
                    self.latest_frame = None
                break
            with self.lock:
                self.latest_frame = frame
            # No sleep: we want to overwrite as fast as frames arrive

    def get_latest(self):
        # Wait until at least one frame is available
        if self.latest_frame is None and self.running:
            # brief wait loop to avoid busy spin at startup
            start = time.time()
            while self.latest_frame is None and self.running and time.time() - start < 2:
                time.sleep(0.005)
        with self.lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
        return frame

    def release(self):
        self.running = False
        try:
            self.reader_thread.join(timeout=1)
        except Exception:
            pass
        self.cap.release()


class DetectionWorker:
    """Runs YOLO object detection on the freshest frame in a background thread.

    Provides near-FPS detection independent of the (slower) depth pipeline.
    The depth loop then fuses the latest detection results with current depth.
    """
    def __init__(self, yolo_model, frame_reader: LatestFrameReader, classes=None, imgsz=288, conf=0.35, interval=0.0):
        self.model = yolo_model
        self.frame_reader = frame_reader
        self.classes = classes if classes is not None else [0, 2, 5, 7]
        self.imgsz = imgsz
        self.conf = conf
        self.interval = max(0.0, interval)
        self.lock = threading.Lock()
        self.latest_results = None
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            frame = self.frame_reader.get_latest()
            if frame is None:
                time.sleep(0.005)
                continue
            try:
                results = self.model(
                    frame,
                    classes=self.classes,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    verbose=False
                )[0]
            except Exception:
                results = None
            with self.lock:
                self.latest_results = results
            if self.interval > 0:
                time.sleep(self.interval)

    def get_latest(self):
        with self.lock:
            return self.latest_results

    def stop(self):
        self.running = False
        try:
            self.thread.join(timeout=1)
        except Exception:
            pass


class DepthWorker:
    """Asynchronous depth inference worker.

    Continuously reads the freshest frame, runs MiDaS + normalization,
    and publishes (depth_array, colored_depth, dmin, dmax, timestamp).
    This decouples heavy depth estimation from the main loop so detection
    alerts can stay low-latency.
    """
    def __init__(self, midas_model, transform_fn, frame_reader: LatestFrameReader, interval: float = 0.0, device=None):
        self.midas = midas_model
        self.transform_fn = transform_fn
        self.frame_reader = frame_reader
        self.interval = max(0.0, interval)
        self.device = device
        self.lock = threading.Lock()
        self.latest_depth = None  # (depth_array, colored_depth, dmin, dmax, ts)
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            frame = self.frame_reader.get_latest()
            if frame is None:
                time.sleep(0.01)
                continue
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                batch = self.transform_fn(rgb).to(self.device)
                with torch.no_grad():
                    prediction = self.midas(batch)
                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=rgb.shape[:2],
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze()
                depth_map = prediction.cpu().numpy()
                depth_min, depth_max = depth_map.min(), depth_map.max()
                norm = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)
                norm = np.clip(norm, 0.0, 1.0)
                colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
                with self.lock:
                    self.latest_depth = (depth_map, colored, depth_min, depth_max, time.time())
            except Exception:
                # Keep previous depth if error
                pass
            if self.interval > 0:
                time.sleep(self.interval)

    def get_latest(self):
        with self.lock:
            return self.latest_depth

    def stop(self):
        self.running = False
        try:
            self.thread.join(timeout=1)
        except Exception:
            pass


class MiDaS:
    # Simple caches to avoid per-frame IO and rendering
    assets_cache = {}
    colorbar_cache = {}
    resized_assets_cache = {}
    # UI state for BirdsEyeView button
    ui_calib_button_rect = None  # (x0, y0, x1, y1)
    ui_calib_request = False     # set True by mouse callback; handled in main loop
    ui_toast_text = None         # short status message rendered in BirdsEyeView
    ui_toast_expire = 0.0        # epoch seconds when toast disappears

    @staticmethod
    def _load_asset(path):
        """Load an image (possibly with alpha) once and cache it."""
        img = MiDaS.assets_cache.get(path)
        if img is None and os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            MiDaS.assets_cache[path] = img
        return img

    @staticmethod
    def _get_resized_asset(path, size):
        """Return a cached resized (w,h) version of the asset (including alpha)."""
        key = (path, size)
        cached = MiDaS.resized_assets_cache.get(key)
        if cached is not None:
            return cached
        base = MiDaS._load_asset(path)
        if base is None:
            return None
        resized = cv2.resize(base, size)
        MiDaS.resized_assets_cache[key] = resized
        return resized

    @staticmethod
    def _get_colorbar(width, height, colormap):
        """Return a cached colorbar image for given size/colormap, including labels."""
        key = (width, height, int(colormap))
        cb = MiDaS.colorbar_cache.get(key)
        if cb is None:
            bar = np.linspace(0, 255, height).astype(np.uint8)[::-1].reshape(-1, 1)
            bar_img = np.repeat(bar, width, axis=1)
            bar_color = cv2.applyColorMap(bar_img, colormap)
            bar_color = bar_color.copy()
            label_positions = [(height-1, 0), ((height-1)//2, 128), (0, 255)]
            for y, val in label_positions:
                label = f"{val}"
                cv2.putText(bar_color, label, (width+5, y+8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            MiDaS.colorbar_cache[key] = bar_color
            cb = bar_color
        return cb.copy()

    @staticmethod
    def _alpha_blit(dst, src, x, y):
        """Blit `src` (BGR or BGRA) onto `dst` at (x,y) with alpha support and clipping.

        - If `src` has 4 channels, uses the 4th as alpha in [0,255].
        - Clamps drawing region to `dst` bounds to avoid index errors.
        - Mutates `dst` in place.
        """
        if dst is None or src is None:
            return
        dh, dw = dst.shape[:2]
        sh, sw = src.shape[:2]
        if sh <= 0 or sw <= 0:
            return
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(dw, x0 + sw)
        y1 = min(dh, y0 + sh)
        if x1 <= x0 or y1 <= y0:
            return
        sx0 = x0 - int(x)
        sy0 = y0 - int(y)
        sx1 = sx0 + (x1 - x0)
        sy1 = sy0 + (y1 - y0)
        src_crop = src[sy0:sy1, sx0:sx1]
        dst_roi = dst[y0:y1, x0:x1]
        if src_crop.shape[2] == 4:
            alpha = (src_crop[:, :, 3] / 255.0)[:, :, None]
            blended = (alpha * src_crop[:, :, :3] + (1.0 - alpha) * dst_roi).astype(np.uint8)
            dst_roi[:] = blended
        else:
            dst_roi[:] = src_crop[:, :, :3]
    @staticmethod
    def _on_birdseye_click(event, x, y, flags, param):
        """OpenCV mouse callback: set a flag when the Calibrate button is clicked."""
        if MiDaS.ui_calib_button_rect is None:
            return
        x0, y0, x1, y1 = MiDaS.ui_calib_button_rect
        in_button = (x0 <= x <= x1 and y0 <= y <= y1)
        # Trigger on both press and release to be forgiving
        if in_button and (event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_LBUTTONUP):
            MiDaS.ui_calib_request = True
    @staticmethod
    def show_birds_eye_view(frame, car_boxes, window_name="BirdsEyeView", yolo_results=None, sensitivity_value=None, show_sensitivity_text: bool = False):
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
        # Create blank white image (Bird's Eye panel)
        view = np.ones((view_h, view_w, 3), dtype=np.uint8) * 255
        # Draw a simple UI button for calibration in the top-right
        btn_w, btn_h = 110, 36
        btn_x = view_w - margin - btn_w
        # Push the button slightly lower so it won't overlap the
        # sensitivity label on compact window sizes
        btn_y = margin + 20
        # Store base (unscaled) button rect first; may be scaled to window size below
        base_btn_rect = (btn_x, btn_y, btn_x + btn_w, btn_y + btn_h)
        cv2.rectangle(view, (btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h), (50, 50, 50), -1)
        cv2.rectangle(view, (btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h), (0, 0, 0), 2)
        cv2.putText(view, "Calibrate", (btn_x + 10, btn_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        # Draw car image in the center bottom
        car_x = view_w // 2 - car_w // 2
        car_y = view_h - car_h - margin
        car_img_path = os.path.join(os.path.dirname(__file__), 'assets', 'birds_eye_view_car.png')
        car_img = MiDaS._get_resized_asset(car_img_path, (car_w, car_h))
        if car_img is not None:
            MiDaS._alpha_blit(view, car_img, car_x, car_y)
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
        if car_boxes:
            arr_boxes = np.array([b[:5] if len(b) == 5 else (*b, 'car') for b in car_boxes], dtype=object)
            centers = (arr_boxes[:,0].astype(float) + arr_boxes[:,2].astype(float)) / 2.0
            w_frame = frame.shape[1]
            zone_indices = np.where(centers < w_frame/3, 0, np.where(centers < 2*w_frame/3, 1, 2))
            for (x1, y1, x2, y2, alert_type), z in zip(arr_boxes, zone_indices):
                zone_colors[z] = (0,0,255)
                zone_alert_types[z] = alert_type
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
        red_car_img = MiDaS._get_resized_asset(red_car_img_path, (car_w, car_h))
        red_person_img = MiDaS._get_resized_asset(red_person_img_path, (car_w, car_h))
        red_bus_img = MiDaS._get_resized_asset(red_bus_img_path, (car_w, car_h))
        green_car_img = MiDaS._get_resized_asset(green_car_img_path, (car_w, car_h))
        green_person_img = MiDaS._get_resized_asset(green_person_img_path, (car_w, car_h))
        green_bus_img = MiDaS._get_resized_asset(green_bus_img_path, (car_w, car_h))
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
                    MiDaS._alpha_blit(view, img_to_use, x_start, y_start)
                    # Draw red border
                    cv2.rectangle(view, (x_start, y_start), (x_start+car_w, y_start+car_h), (0,0,255), 4)
                elif car_img is not None:
                    MiDaS._alpha_blit(view, car_img, x_start, y_start)
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
                    MiDaS._alpha_blit(view, img_to_use, x_start, y_start)
        # Draw car image again at the bottom (your car)
        if car_img is not None:
            MiDaS._alpha_blit(view, car_img, car_x, car_y)
        else:
            cv2.rectangle(view, (car_x, car_y), (car_x + car_w, car_y + car_h), (50, 50, 50), -1)
        # Optional: draw sensitivity value text in the panel (off by default)
        if show_sensitivity_text and sensitivity_value is not None:
            text = f"Sensitivity: {sensitivity_value}"
            cv2.putText(view, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)
        # Optional toast: transient on-screen message (e.g., calibration results)
        now = time.time()
        if MiDaS.ui_toast_text and now < MiDaS.ui_toast_expire:
            cv2.rectangle(view, (10, 10), (10 + 200, 10 + 30), (0, 0, 0), -1)
            cv2.putText(view, MiDaS.ui_toast_text, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        # Scale the view to the current window size so the user can resize freely,
        # and update the button hitbox to match the scaled coordinates.
        scaled_view = view
        try:
            _x, _y, win_w, win_h = cv2.getWindowImageRect(window_name)
            if win_w > 0 and win_h > 0 and (win_w != view_w or win_h != view_h):
                scale_x = win_w / float(view_w)
                scale_y = win_h / float(view_h)
                scaled_view = cv2.resize(view, (win_w, win_h), interpolation=cv2.INTER_LINEAR)
                bx0, by0, bx1, by1 = base_btn_rect
                MiDaS.ui_calib_button_rect = (
                    int(bx0 * scale_x), int(by0 * scale_y), int(bx1 * scale_x), int(by1 * scale_y)
                )
            else:
                MiDaS.ui_calib_button_rect = base_btn_rect
        except Exception:
            MiDaS.ui_calib_button_rect = base_btn_rect
        cv2.imshow(window_name, scaled_view)
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
        """Normalizes and colorizes a depth map for visualization. Returns colorized map and min/max values.
        Uses epsilon and clipping for numerical stability to match vectorized checks."""
        depth_min, depth_max = depth_map.min(), depth_map.max()
        denom = depth_max - depth_min
        # Add epsilon to avoid divide-by-zero if scene has near-constant depth
        normalized = (depth_map - depth_min) / (denom + 1e-8)
        normalized = np.clip(normalized, 0.0, 1.0)
        normalized_u8 = (normalized * 255).astype(np.uint8)
        colorized = cv2.applyColorMap(normalized_u8, cv2.COLORMAP_INFERNO)
        return colorized, depth_min, depth_max

    @staticmethod
    def draw_depth_colorbar(image, vmin, vmax, colormap=cv2.COLORMAP_INFERNO, width=30, height=200, margin=10):
        """Draws a vertical colorbar on the right side of the image showing the depth range."""
        bar_color = MiDaS._get_colorbar(width, height, colormap)
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

    # Removed unused helper `stack_frames` (no references). Keeping logic inline
    # in the main loop avoids unnecessary extra copies and keeps behavior identical.

    def infer_video(self, source: str = 0, output_path: str = None, display_main: bool = True, display_birdseye: bool = True, sound_enabled: bool = True, latest_only: bool = True, parallel_detection: bool = True, detection_interval: float = 0.0, detection_imgsz: int = 256, parallel_depth: bool = True, depth_interval: float = 0.0) -> None:
        """Performs real-time depth estimation on a video stream.

        Args:
            source: Camera index or video file path.
            output_path: Optional path to save combined output video.
            display_main: Show the combined LDW + depth window.
            display_birdseye: Show Bird's Eye proximity window (with sensitivity trackbar).
            sound_enabled: Enable audio alerts even if windows are hidden.
        """
        width = 1280
        height = 640

        # Decide whether the source is a file or a stream/camera.
        # Rationale:
        # - Streams/cameras benefit from a LatestFrameReader to skip backlog and reduce latency.
        # - Files must be read sequentially (cv2.VideoCapture.read()) to avoid freezing at EOF
        #   and to ensure deterministic frame-by-frame processing.
        is_stream = False
        if isinstance(source, (int, float)):
            is_stream = True
        elif isinstance(source, str):
            if source.isdigit():
                is_stream = True
            else:
                low = source.lower()
                if low.startswith("rtsp://") or low.startswith("http://") or low.startswith("https://"):
                    is_stream = True
                else:
                    is_stream = False  # treat as file path

        # Use latest-only reader only for streams; files are read sequentially to avoid EOF freeze
        use_reader = latest_only and is_stream

        if use_reader:
            reader = LatestFrameReader(source)
            fps = reader.cap.get(cv2.CAP_PROP_FPS) or 30
        else:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video source: {source}")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        transform = self.transforms()

        # Load YOLO model for object detection (enable half precision if GPU)
        yolo_model = YOLO("yolo11n.pt")
        if self.device.type == 'cuda':
            try:
                yolo_model.to(self.device)
                yolo_model.model.half()
            except Exception:
                pass
        class_names = yolo_model.model.names if hasattr(yolo_model.model, 'names') else [str(i) for i in range(100)]
        # Background detection worker is enabled only when using the reader (i.e., streams),
        # to avoid mixing async results with sequential file decoding.
        if parallel_detection and use_reader:
            detection_worker = DetectionWorker(
                yolo_model,
                reader,
                classes=[0, 2, 5, 7],
                imgsz=detection_imgsz,
                conf=0.35,
                interval=detection_interval
            )
        else:
            detection_worker = None

        # Background depth worker follows the same rule as detection.
        if parallel_depth and use_reader:
            depth_worker = DepthWorker(self.midas, transform, reader, interval=depth_interval, device=self.device)
        else:
            depth_worker = None

        print("🚀 Starting video depth inference... Press 'q' to quit.")
        # If we're displaying UI, create the BirdsEyeView window and a sensitivity
        # trackbar so the user can tune the Inferno colormap cutoff (0-255).
        if display_birdseye:
            try:
                cv2.namedWindow("BirdsEyeView", cv2.WINDOW_NORMAL)
                # Allow free aspect ratio and start with a comfortable size
                cv2.setWindowProperty("BirdsEyeView", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FREERATIO)
                cv2.resizeWindow("BirdsEyeView", 360, 520)
                cv2.createTrackbar('Sensitivity', 'BirdsEyeView', BIRDSEYE_SENSITIVITY_DEFAULT, 255, lambda x: None)
                cv2.setMouseCallback("BirdsEyeView", MiDaS._on_birdseye_click)
            except Exception:
                pass
        # Make the main window resizable like a normal window and start at our base size
        main_win_name = "MiDaS Depth Estimation (Press 'q' to exit)"
        if display_main:
            try:
                cv2.namedWindow(main_win_name, cv2.WINDOW_NORMAL)
                cv2.setWindowProperty(main_win_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FREERATIO)
                cv2.resizeWindow(main_win_name, width, height)
            except Exception:
                pass

        while True:
            # Obtain the freshest frame (skip any backlog)
            if use_reader:
                frame = reader.get_latest()
                # If the reader hit EOF or error, it sets latest_frame to None; exit
                if frame is None:
                    break
            else:
                ret, frame = cap.read()
                if not ret:
                    break

            # Acquire latest depth (async) or compute synchronously if disabled
            if depth_worker is not None:
                depth_payload = depth_worker.get_latest()
                if depth_payload is None:
                    depth = None
                    colored_depth_with_boxes = np.zeros_like(frame)
                    dmin = dmax = 0.0
                    normalized_uint8 = None
                else:
                    depth, colored_depth, dmin, dmax, _ts = depth_payload
                    colored_depth_with_boxes = colored_depth.copy()
                    norm = (depth - dmin) / (dmax - dmin + 1e-8)
                    normalized_uint8 = (np.clip(norm, 0.0, 1.0) * 255).astype(np.uint8)
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_batch = transform(rgb).to(self.device)
                depth = self.depth_map(input_batch, rgb)
                colored_depth, dmin, dmax = self.normalize_depth(depth)
                colored_depth_with_boxes = colored_depth.copy()
                norm = (depth - dmin) / (dmax - dmin + 1e-8)
                normalized_uint8 = (np.clip(norm, 0.0, 1.0) * 255).astype(np.uint8)


            # --- Object Detection and Alert Logic ---
            if parallel_detection and detection_worker is not None:
                results = detection_worker.get_latest()
                if results is not None and results.boxes is not None:
                    all_class_ids = results.boxes.cls.cpu().numpy().astype(int)
                    boxes_all = results.boxes.xyxy.cpu().numpy()
                else:
                    all_class_ids = []
                    boxes_all = np.empty((0, 4))
            else:
                results = yolo_model(
                    frame,
                    classes=[0, 2, 5, 7],  # person, car, bus, truck
                    imgsz=288,
                    conf=0.35,
                    verbose=False
                )[0]
                if results.boxes is not None:
                    all_class_ids = results.boxes.cls.cpu().numpy().astype(int)
                    boxes_all = results.boxes.xyxy.cpu().numpy()
                else:
                    all_class_ids = []
                    boxes_all = np.empty((0, 4))

            alert_triggered = False
            alert_texts = set()
            alert_indices = set()
            # Read sensitivity cutoff from the trackbar (if available) and compute
            # the allowed BGR set for alerting. The cutoff is the lower colormap
            # index considered 'close' (higher index = closer in Inferno mapping).
            if display_birdseye:
                try:
                    cutoff = int(cv2.getTrackbarPos('Sensitivity', 'BirdsEyeView'))
                except Exception:
                    cutoff = BIRDSEYE_SENSITIVITY_DEFAULT
            else:
                cutoff = BIRDSEYE_SENSITIVITY_DEFAULT

            cutoff = max(0, min(255, cutoff))

            # Precompute depth proximity only if depth exists
            if normalized_uint8 is not None:
                close_map = (normalized_uint8 >= cutoff).astype(np.uint8)
                integral = cv2.integral(close_map, sdepth=cv2.CV_32S)
            else:
                close_map = None
                integral = None

            def region_sum(integral_img, x1, y1, x2, y2):
                """Sum in inclusive rectangle using cv2.integral output (with offset)."""
                # cv2.integral has shape (h+1, w+1); shift coordinates by +1
                A = integral_img[y2+1, x2+1]
                B = integral_img[y1, x2+1]
                C = integral_img[y2+1, x1]
                D = integral_img[y1, x1]
                return A - B - C + D

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
                # O(1) close-range pixel count via integral image (inclusive bounds)
                if x2 <= x1 or y2 <= y1:
                    continue
                if integral is not None:
                    match_count = region_sum(integral, x1, y1, x2-1, y2-1)
                    total_pixels = (x2 - x1) * (y2 - y1)
                    color_match_ratio = match_count / total_pixels
                    is_alert = color_match_ratio > 0.75
                else:
                    is_alert = False  # Depth not ready yet; only awareness
                # Commented below to save memory
                # print(f"Object: {class_names[cls_id] if cls_id < len(class_names) else str(cls_id)}, match: {match_count}, total: {total_pixels}, color_match_ratio: {color_match_ratio:.2f}, alert: {is_alert}")
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
                if idx in alert_indices:
                    color = (0, 0, 255)  # Red for alert (car or pedestrian)
                else:
                    color = (0, 255, 0) if cls_id == 2 else (255, 255, 0)
                label = f"{class_names[cls_id] if cls_id < len(class_names) else str(cls_id)}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.rectangle(colored_depth_with_boxes, (x1, y1), (x2, y2), color, 2)
                cv2.putText(colored_depth_with_boxes, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if alert_triggered and sound_enabled:
                threading.Thread(target=play_alert_once, daemon=True).start()
                for i, text in enumerate(sorted(alert_texts)):
                    if display_main:
                        cv2.putText(frame, text, (30, 60 + i * 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

            # Add colorbar after drawing boxes so it is always on top
            if depth is not None:
                colored_depth_with_boxes = self.draw_depth_colorbar(colored_depth_with_boxes, dmin, dmax)


            # Lane Departure Warning overlay (left side)
            ldw_frame = ldw_overlay(frame)
            # Draw the guide line and text on the LDW overlay only (not on the frame used for detection)
            guide_y = int(ldw_frame.shape[0] * 0.95)
            cv2.line(ldw_frame, (0, guide_y), (ldw_frame.shape[1], guide_y), (0, 255, 255), 2)
            cv2.putText(ldw_frame, 'Align bonnet here (BELOW NOT DETECTED)', (10, guide_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            # Resize LDW overlay and depth map to same size
            ldw_frame_resized = cv2.resize(ldw_frame, (width // 2, height))
            depth_resized = cv2.resize(colored_depth_with_boxes, (width // 2, height))
            # Concatenate LDW overlay (left) and depth map (right)
            combined = np.hstack((ldw_frame_resized, depth_resized))

            if display_main:
                # Scale the combined image to the current window size so resizing works naturally
                render_img = combined
                try:
                    _x, _y, win_w, win_h = cv2.getWindowImageRect(main_win_name)
                    if win_w > 0 and win_h > 0 and (win_w != combined.shape[1] or win_h != combined.shape[0]):
                        render_img = cv2.resize(combined, (win_w, win_h), interpolation=cv2.INTER_LINEAR)
                except Exception:
                    pass
                cv2.imshow(main_win_name, render_img)
            if display_birdseye:
                self.show_birds_eye_view(frame, car_boxes_birdseye, yolo_results=results, sensitivity_value=cutoff)

            if writer:
                writer.write(combined)

            # Helper: run calibration (used by key 'c' and the BirdsEyeView button).
            # It picks the nearest detected object (highest median proximity),
            # then sets the Sensitivity trackbar to the 25th percentile of that ROI so ~75% of
            # its pixels count as "close" in the Inferno colormap.
            def _run_calibration():
                try:
                    if normalized_uint8 is None or len(boxes_all) == 0:
                        print("[CAL] No depth or detections available for calibration.")
                        MiDaS.ui_toast_text = "No data yet"
                        MiDaS.ui_toast_expire = time.time() + 1.5
                        return
                    guide_y_local = int(frame.shape[0] * 0.95)
                    best_idx = -1
                    best_med = -1.0
                    for idx, (box, cls_id) in enumerate(zip(boxes_all, all_class_ids)):
                        if cls_id not in [0, 2, 5, 7]:
                            continue
                        x1, y1, x2, y2 = map(int, box)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1]-1, x2), min(frame.shape[0]-1, y2)
                        if y1 > guide_y_local:
                            continue
                        if x2 <= x1 or y2 <= y1:
                            continue
                        roi = normalized_uint8[y1:y2, x1:x2]
                        if roi.size < 50:
                            continue
                        med = float(np.median(roi))
                        if med > best_med:
                            best_med = med
                            best_idx = idx
                    if best_idx == -1:
                        print("[CAL] No suitable object ROI for calibration.")
                        return
                    x1, y1, x2, y2 = map(int, boxes_all[best_idx])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1]-1, x2), min(frame.shape[0]-1, y2)
                    roi = normalized_uint8[y1:y2, x1:x2].astype(np.float32)
                    recommended = int(np.clip(np.percentile(roi, 25.0), 0, 255))
                    if display_birdseye:
                        try:
                            cv2.setTrackbarPos('Sensitivity', 'BirdsEyeView', recommended)
                        except Exception:
                            # If the trackbar isn't available (shouldn't happen when clicking the button),
                            # just fall back to showing a toast.
                            pass
                    print(f"[CAL] Set sensitivity to {recommended} (ROI median={best_med:.1f}).")
                    MiDaS.ui_toast_text = f"Set: {recommended}"
                    MiDaS.ui_toast_expire = time.time() + 1.8
                except Exception as e:
                    print(f"[CAL] Calibration error: {e}")
                    MiDaS.ui_toast_text = "Calib error"
                    MiDaS.ui_toast_expire = time.time() + 1.5

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('c'):
                _run_calibration()

            # UI button click handling (works for camera, IP stream, and files)
            if MiDaS.ui_calib_request:
                _run_calibration()
                MiDaS.ui_calib_request = False

        if use_reader:
            reader.release()
        else:
            cap.release()
        if detection_worker is not None:
            detection_worker.stop()
        if depth_worker is not None:
            depth_worker.stop()
        if writer:
            writer.release()
        if display_main or display_birdseye:
            cv2.destroyAllWindows()
        print("✅ Inference completed and resources released.")



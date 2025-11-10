import cv2
import torch
import numpy as np
import warnings
from ultralytics import YOLO
import os
warnings.filterwarnings("ignore", category=FutureWarning)


class MiDaS:
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
        # Add value labels
        bar_color = bar_color.copy()
        for i, val in enumerate([vmin, (vmin+vmax)/2, vmax]):
            y = int(i * (height-1) / 2)
            label = f"{val:.{decimals}f}"
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




            # --- Object Detection ---
            results = yolo_model(frame)[0]
            if results.boxes is not None:
                all_class_ids = results.boxes.cls.cpu().numpy().astype(int)
                boxes_all = results.boxes.xyxy.cpu().numpy()
            else:
                all_class_ids = []
                boxes_all = np.empty((0, 4))

            # Only consider 'person' and 'car' (COCO: person=0, car=2)
            alert_triggered = False
            alert_text = ""
            ALERT_DEPTH_THRESHOLD = 400  # Set to match depth scale (100-500)
            alert_indices = set()
            for idx, (box, cls_id) in enumerate(zip(boxes_all, all_class_ids)):
                if cls_id not in [0, 2]:
                    continue
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1]-1, x2), min(frame.shape[0]-1, y2)
                depth_region = depth[y1:y2, x1:x2]
                if depth_region.size > 0:
                    mean_depth = float(np.mean(depth_region))
                    min_depth = float(np.min(depth_region))
                else:
                    mean_depth = float('nan')
                    min_depth = float('nan')
                print(f"Object: {class_names[cls_id] if cls_id < len(class_names) else str(cls_id)}, min_depth: {min_depth:.4f}, mean_depth: {mean_depth:.4f}")
                # Alert if any part of object is close (min depth)
                is_alert = (not np.isnan(min_depth)) and (min_depth < ALERT_DEPTH_THRESHOLD)
                if is_alert:
                    alert_triggered = True
                    alert_indices.add(idx)
                    if cls_id == 2:
                        alert_text = "Collision Warning! (Car)"
                    elif cls_id == 0:
                        alert_text = "Pedestrian Alert!"
                # Draw all boxes, highlight alert ones in red
                for idx, (box, cls_id) in enumerate(zip(boxes_all, all_class_ids)):
                    if cls_id not in [0, 2]:
                        continue
                    x1, y1, x2, y2 = map(int, box)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1]-1, x2), min(frame.shape[0]-1, y2)
                    depth_region = depth[y1:y2, x1:x2]
                    if depth_region.size > 0:
                        mean_depth = float(np.mean(depth_region))
                    else:
                        mean_depth = float('nan')
                    if idx in alert_indices:
                        color = (0, 0, 255)  # Red for alert
                    else:
                        color = (0, 255, 0) if cls_id == 2 else (255, 255, 0)
                    label = f"{class_names[cls_id] if cls_id < len(class_names) else str(cls_id)}: {mean_depth:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.rectangle(colored_depth_with_boxes, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(colored_depth_with_boxes, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if alert_triggered:
                cv2.putText(frame, alert_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

            # Add colorbar after drawing boxes so it is always on top
            colored_depth_with_boxes = self.draw_depth_colorbar(colored_depth_with_boxes, dmin, dmax)

            # Stack the annotated frame and the annotated depth map
            combined = self.stack_frames(frame, colored_depth_with_boxes, width, height)

            if display:
                cv2.imshow("MiDaS Depth Estimation (Press 'q' to exit)", combined)

            if writer:
                writer.write(combined)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("✅ Inference completed and resources released.")


if __name__ == "__main__":
    """Example usage for both image and video depth estimation."""
    # midas = MiDaS(model_type="MiDaS_small")
    midas = MiDaS(model_type="DPT_Hybrid")
    # midas = MiDaS(model_type="DPT_Large")
    

    # ---------- Video inference ----------
    # For webcam: source=0
    # For file:   source="cars.mp4"
    midas.infer_video(source="pedestrian_crash.mp4", output_path="depth_video.mp4", display=True)
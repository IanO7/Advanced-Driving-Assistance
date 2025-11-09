def grey(image):
    image = np.asarray(image)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def gauss(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def canny(image):
    edges = cv2.Canny(image, 50, 150)
    return edges

def region(image):
    height, width = image.shape
    # Widen and lower the triangle for wide roads
    triangle = np.array([
        [(int(width*0.05), height), (int(width*0.5), int(height*0.55)), (int(width*0.95), height)]
    ])
    mask = np.zeros_like(image)
    mask = cv2.fillPoly(mask, triangle, 255)
    mask = cv2.bitwise_and(image, mask)
    return mask

def average(image, lines):
    import warnings
    left = []
    right = []
    if lines is None:
        return None
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', np.RankWarning)
            try:
                parameters = np.polyfit((x1, x2), (y1, y2), 1)
            except Exception:
                continue
        slope = parameters[0]
        y_int = parameters[1]
        if slope < 0:
            left.append((slope, y_int))
        else:
            right.append((slope, y_int))
    left_line = make_points(image, np.average(left, axis=0)) if len(left) >= 2 else None
    right_line = make_points(image, np.average(right, axis=0)) if len(right) >= 2 else None
    return np.array([left_line, right_line])

def make_points(image, average):
    slope, y_int = average
    y1 = image.shape[0]
    y2 = int(image.shape[0] * 0.55)  # match triangle top
    if slope == 0:
        slope = 0.1
    x1 = int((y1 - y_int) // slope)
    x2 = int((y2 - y_int) // slope)
    return np.array([x1, y1, x2, y2])

def display_lines(image, lines):
    lines_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            if line is not None:
                x1, y1, x2, y2 = line
                cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return lines_image

import cv2
import numpy as np
import edge_detection as edge

class Lane:
    def __init__(self, orig_frame):
        self.orig_frame = orig_frame
        self.lane_line_markings = None
        self.warped_frame = None
        self.transformation_matrix = None
        self.inv_transformation_matrix = None
        self.orig_image_size = self.orig_frame.shape[::-1][1:]
        width = self.orig_image_size[0]
        height = self.orig_image_size[1]
        # You may need to tune these points for your video
        self.roi_points = np.float32([
            (int(width*0.45), int(height*0.6)),
            (int(width*0.1), height),
            (int(width*0.9), height),
            (int(width*0.55), int(height*0.6))
        ])
        self.padding = int(0.25 * width)
        self.desired_roi_points = np.float32([
            [self.padding, 0],
            [self.padding, self.orig_image_size[1]],
            [self.orig_image_size[0]-self.padding, self.orig_image_size[1]],
            [self.orig_image_size[0]-self.padding, 0]
        ])
        self.no_of_windows = 10
        self.margin = int((1/12) * width)
        self.minpix = int((1/24) * width)
        self.left_fit = None
        self.right_fit = None
        self.ploty = None
        self.left_fitx = None
        self.right_fitx = None
        self.leftx = None
        self.rightx = None
        self.lefty = None
        self.righty = None

    def get_line_markings(self, frame=None):
        if frame is None:
            frame = self.orig_frame
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        _, sxbinary = edge.threshold(hls[:, :, 1], thresh=(120, 255))
        sxbinary = edge.blur_gaussian(sxbinary, ksize=3)
        sxbinary = edge.mag_thresh(sxbinary, sobel_kernel=3, thresh=(110, 255))
        s_channel = hls[:, :, 2]
        _, s_binary = edge.threshold(s_channel, (80, 255))
        _, r_thresh = edge.threshold(frame[:, :, 2], thresh=(120, 255))
        rs_binary = cv2.bitwise_and(s_binary, r_thresh)
        self.lane_line_markings = cv2.bitwise_or(rs_binary, sxbinary.astype(np.uint8))
        return self.lane_line_markings

    def perspective_transform(self, frame=None):
        if frame is None:
            frame = self.lane_line_markings
        self.transformation_matrix = cv2.getPerspectiveTransform(self.roi_points, self.desired_roi_points)
        self.inv_transformation_matrix = cv2.getPerspectiveTransform(self.desired_roi_points, self.roi_points)
        self.warped_frame = cv2.warpPerspective(frame, self.transformation_matrix, self.orig_image_size, flags=(cv2.INTER_LINEAR))
        (thresh, binary_warped) = cv2.threshold(self.warped_frame, 127, 255, cv2.THRESH_BINARY)
        self.warped_frame = binary_warped
        return self.warped_frame

    def calculate_histogram(self, frame=None):
        if frame is None:
            frame = self.warped_frame
        self.histogram = np.sum(frame[int(frame.shape[0]/2):,:], axis=0)
        return self.histogram

    def histogram_peak(self):
        midpoint = np.int32(self.histogram.shape[0]/2)
        leftx_base = np.argmax(self.histogram[:midpoint])
        rightx_base = np.argmax(self.histogram[midpoint:]) + midpoint
        return leftx_base, rightx_base

    def get_lane_line_indices_sliding_windows(self):
        margin = self.margin
        window_height = np.int32(self.warped_frame.shape[0]/self.no_of_windows)
        nonzero = self.warped_frame.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        left_lane_inds = []
        right_lane_inds = []
        leftx_base, rightx_base = self.histogram_peak()
        leftx_current = leftx_base
        rightx_current = rightx_base
        for window in range(self.no_of_windows):
            win_y_low = self.warped_frame.shape[0] - (window + 1) * window_height
            win_y_high = self.warped_frame.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            if len(good_left_inds) > self.minpix:
                leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        self.leftx = leftx
        self.rightx = rightx
        self.lefty = lefty
        self.righty = righty
        self.left_fit = np.polyfit(lefty, leftx, 2) if len(leftx) > 0 and len(lefty) > 0 else None
        self.right_fit = np.polyfit(righty, rightx, 2) if len(rightx) > 0 and len(righty) > 0 else None
        return self.left_fit, self.right_fit

    def get_lane_overlay(self):
        if self.left_fit is None or self.right_fit is None:
            return self.orig_frame
        ploty = np.linspace(0, self.warped_frame.shape[0]-1, self.warped_frame.shape[0])
        left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
        right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]
        warp_zero = np.zeros_like(self.warped_frame).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))] )
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(color_warp, np.int32([pts]), (0,255, 0))
        newwarp = cv2.warpPerspective(color_warp, self.inv_transformation_matrix, (self.orig_frame.shape[1], self.orig_frame.shape[0]))
        result = cv2.addWeighted(self.orig_frame, 1, newwarp, 0.3, 0)
        return result

from ultralytics import solutions
cap = cv2.VideoCapture("car_crash.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("distance_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
distancecalculator = solutions.DistanceCalculation(
    model="yolo11n.pt",
    show=False,
)
reference_idx = None
last_results = None

# Calculate the closest point on the box to a given reference point
def get_closest_edge_point(box, ref_point):
    x1, y1, x2, y2 = box
    rx, ry = ref_point
    # Clamp the reference point to the box bounds
    closest_x = min(max(rx, x1), x2)
    closest_y = min(max(ry, y1), y2)
    return closest_x, closest_y
def mouse_callback(event, x, y, flags, param):
    global reference_idx, last_boxes
    if reference_idx is not None:
        return
    if event == cv2.EVENT_LBUTTONDOWN and last_boxes is not None and len(last_boxes) > 0:
        found = False
        for idx, box in enumerate(last_boxes):
            x1, y1, x2, y2 = box
            if x1 <= x <= x2 and y1 <= y <= y2:
                reference_idx = idx
                print(f"Reference object selected: {idx} at ({x}, {y})")
                found = True
                break
        if not found:
            print(f"Clicked at ({x}, {y}) but no object was selected.")
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
reference_idx = None
last_boxes = None
cv2.namedWindow("Distances")
cv2.setMouseCallback("Distances", mouse_callback)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or processing is complete.")
        break

    # Resize frame to 1920x1080 for standardization
    im0 = cv2.resize(im0, (1920, 1080))

    # Run YOLO detection
    results = model(im0)[0]
    # Only keep boxes for 'car' and 'person' classes (COCO: car=2, person=0)
    if results.boxes is not None:
        all_class_ids = results.boxes.cls.cpu().numpy().astype(int)
        boxes_all = results.boxes.xyxy.cpu().numpy()
        filtered = [(box, cls_id) for box, cls_id in zip(boxes_all, all_class_ids) if cls_id in [0, 2]]
        boxes = np.array([b for b, _ in filtered]) if filtered else np.empty((0, 4))
        class_ids = [cls_id for _, cls_id in filtered]
    else:
        boxes = np.empty((0, 4))
        class_ids = []
    last_boxes = boxes

    plot_im = im0.copy()
    # Define reference point for distance calculations (just above the bottom center of the screen, ~1cm from bottom)
    frame_center_x = plot_im.shape[1] // 2
    # Move the reference point to about 92% of the frame height (a bit higher above the bottom)
    frame_ref_y = int(plot_im.shape[0] * 0.92)

    # --- Lane detection (Automatic Addison pipeline) ---
    lane_obj = Lane(plot_im)
    lane_obj.get_line_markings()
    lane_obj.perspective_transform()
    lane_obj.calculate_histogram()
    lane_obj.get_lane_line_indices_sliding_windows()
    plot_im = lane_obj.get_lane_overlay()

    # --- Car and pedestrian detection logic ---
    # Draw the reference point (bottom center of the screen) used for distance calculations
    ref_point = (frame_center_x, frame_ref_y)
    
    cv2.circle(plot_im, ref_point, 10, (0, 0, 255), -1)  # Red filled circle being drawn
    # ALERT DISTANCES ARE MEASURED FROM THE CENTROID OF THE FRONT CAR (OR PEDESTRIAN) TO THE REFERENCE POINT (red dot, 80% down the screen).
    # This ensures only objects directly in front and close to the vehicle trigger alerts, matching real-world collision risk.
    COLLISION_THRESHOLD = 200  # Pixel distance for collision warning
    PEDESTRIAN_THRESHOLD = 335  # Pixel distance for pedestrian alert
    collision_indices = set()
    pedestrian_indices = set()
    # Mark cars and pedestrians close to the bottom-center reference point
    if len(boxes) > 0:
        for idx, (box, cls_id) in enumerate(zip(boxes, class_ids)):
            ref_pt = (frame_center_x, frame_ref_y)
            closest = get_closest_edge_point(box, ref_pt)
            # Ignore objects below the red reference point (i.e., in the bonnet area)
            if closest[1] > frame_ref_y:
                continue
            # Use only the vertical (y-axis) distance
            dist = abs(frame_ref_y - closest[1])
            # **TEST DISTANCE ONLY!
            print(f"Object {idx} (class {cls_id}): vertical pixel distance to ref point = {dist}")
            if cls_id == 2:  # car
                if dist < COLLISION_THRESHOLD:
                    collision_indices.add(idx)
            elif cls_id == 0:  # person
                if dist < PEDESTRIAN_THRESHOLD:
                    pedestrian_indices.add(idx)

    # ... lane detection (main lines already drawn above) ...

    for idx, (box, cls_id) in enumerate(zip(boxes, class_ids)):
        x1, y1, x2, y2 = map(int, box)
        # Car coloring
        if cls_id == 2:
            if idx in collision_indices:
                color = (0, 0, 255)  # Red for collision (close to bottom-center reference point)
            else:
                color = (0, 255, 0)  # Green for all other cars
            label = f"{idx} car"
        # Pedestrian coloring
        elif cls_id == 0:
            if idx in pedestrian_indices:
                color = (255, 0, 255)  # Magenta for close pedestrian (to bottom-center reference point)
            else:
                color = (255, 255, 0)  # Cyan for other pedestrians
            label = f"{idx} person"
        else:
            color = (200, 200, 200)
            label = f"{idx} obj"
        cv2.rectangle(plot_im, (x1, y1), (x2, y2), color, 2)
        cv2.putText(plot_im, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Display pedestrian warning if any close pedestrian detected
    if pedestrian_indices:
        cv2.putText(plot_im, "Pedestrian Alert!", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 4)

    # Display collision warning text if collision detected for front car
    if collision_indices:
        cv2.putText(plot_im, "Collision Warning!", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    # Draw distances if reference object is selected
    if reference_idx is not None and len(boxes) > 0 and reference_idx < len(boxes):
        ref_pt = (frame_center_x, frame_ref_y)
        ref_edge = get_closest_edge_point(boxes[reference_idx], ref_pt)
        for idx, box in enumerate(boxes):
            if idx == reference_idx:
                continue
            edge = get_closest_edge_point(box, ref_pt)
            dist = abs(ref_edge[1] - edge[1])
            # Draw only the vertical distance value (no line)
            mid_pt = ((ref_edge[0] + edge[0]) // 2, (ref_edge[1] + edge[1]) // 2)
            cv2.putText(plot_im, f"{dist:.1f}px", mid_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Distances", plot_im)
    video_writer.write(plot_im)

    key = cv2.waitKey(1)
    if key == 27:  # ESC to exit
        break
    # Remove the ability to reset the reference object after initial selection

cap.release()
video_writer.release()
cv2.destroyAllWindows()
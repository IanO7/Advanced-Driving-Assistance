"""Lane Detection (LDW Overlay) Module.

Rewritten to use the provided pipeline implementation while keeping the
public API identical (the function `ldw_overlay(frame)` returns a BGR image
with lane markings overlayed). Depth estimation and main program imports
remain unaffected.

Pipeline summary:
1. ROI masking (quadrilateral)
2. Gaussian blur
3. Canny edge detection
4. Probabilistic Hough transform
5. Averaging left/right lane segments into solid lines
6. Optional polygon fill between lanes
"""

import cv2
import numpy as np

def region_of_interest(img, vertices):
    """Apply a mask to keep only the region of interest (ROI)."""
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    return cv2.bitwise_and(img, mask)

def draw_lines(img, lines, color=[0, 255, 0], thickness=5):
    """Draw single averaged left/right lane lines and softly fill polygon."""
    img[:] = 0  # clear before drawing
    left_line = None
    right_line = None
    if lines is not None:
        left_lines = []
        right_lines = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2 - x1 == 0:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < 0.3:  # skip nearly horizontal
                    continue
                if slope < 0:
                    left_lines.append((x1, y1, x2, y2))
                else:
                    right_lines.append((x1, y1, x2, y2))

        def make_line(points, y_min, y_max):
            if len(points) == 0:
                return None
            x_coords = []
            y_coords = []
            for x1, y1, x2, y2 in points:
                x_coords += [x1, x2]
                y_coords += [y1, y2]
            poly = np.polyfit(y_coords, x_coords, 1)  # x = m*y + b
            x_start = int(poly[0] * y_max + poly[1])
            x_end = int(poly[0] * y_min + poly[1])
            return (x_start, y_max, x_end, y_min)

        height = img.shape[0]
        y_min = int(height * 0.6)
        y_max = height
        left_line = make_line(left_lines, y_min, y_max)
        right_line = make_line(right_lines, y_min, y_max)

        if left_line is not None and right_line is not None:
            polygon_points = np.array([
                [left_line[0], left_line[1]],
                [left_line[2], left_line[3]],
                [right_line[2], right_line[3]],
                [right_line[0], right_line[1]]
            ], dtype=np.int32)
            overlay = img.copy()
            cv2.fillPoly(overlay, [polygon_points], color=(180, 220, 255))
            cv2.addWeighted(overlay, 0.4, img, 0.6, 0, dst=img)

    for line in [left_line, right_line]:
        if line is not None:
            x1, y1, x2, y2 = line
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def _pipeline(image):
    """Internal pipeline returning original image with lane overlay."""
    # NOTE: The provided code uses RGB->GRAY but frames are BGR from OpenCV.
    # We keep the original constant from the snippet for fidelity; adjust to
    # COLOR_BGR2GRAY if color weighting consistency is required.
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    cannyed_image = cv2.Canny(blurred, 50, 150)
    height, width = image.shape[:2]
    roi_vertices = [
        (int(width * 0.1), int(height * 0.95)),
        (int(width * 0.4), int(height * 0.6)),
        (int(width * 0.6), int(height * 0.6)),
        (int(width * 0.9), int(height * 0.95))
    ]
    cropped_image = region_of_interest(cannyed_image, np.array([roi_vertices], np.int32))
    lines = cv2.HoughLinesP(
        cropped_image,
        rho=6,
        theta=np.pi / 60,
        threshold=160,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=25
    )
    line_image = np.zeros_like(image)
    draw_lines(line_image, lines)
    final_image = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    return final_image

def ldw_overlay(frame):
    """Public API: apply lane detection overlay to a BGR frame and return BGR frame."""
    return _pipeline(frame)

__all__ = ["ldw_overlay"]

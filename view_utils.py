from typing import List

import cv2
import numpy as np


name_to_color = {
    'car': (0., 0., 1),
    'person': (1., 0., 0),
    'traffic light': (0., 1., 1.),
    'bicycle': (0., 0.5, 1.),
    'truck': (0.2, 0., 0.4),
    'bus': (0.5, 0., 1.),
    'motorbike': (1.0, 0.4, 0.4),
    'bench': (1., 1., 0.4)
}


def blend(img1, img2, alpha=0.5):
    beta = 1 - alpha
    out = np.uint8(img1 * alpha + img2 * beta)
    return out


def draw_detections(frame: np.ndarray,
                    birdeye: np.ndarray,
                    postprocessed_detections: List[dict],
                    name_to_color: dict,
                    line_thickness: int = 3):

    for d in postprocessed_detections:

        try:
            color = name_to_color[d['name']]
        except KeyError:
            print(d['name'])
            color = (1., 1., 1.)
        color = [int(c * 255) for c in color]

        xmin, ymin, xmax, ymax = d['coords']

        # Draw bounding box
        frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color,
                              thickness=line_thickness)

        # Draw text of class name
        text = d['name']
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.
        font_thickness = 2
        (label_w, label_h), baseline = cv2.getTextSize(text, font,
                                                       fontScale=font_scale,
                                                       thickness=font_thickness)
        x_start = xmin - line_thickness // 2
        cv2.rectangle(frame, (x_start, ymin), (x_start+label_w, ymin + label_h),
                      color=color, thickness=cv2.FILLED)
        cv2.putText(frame, text, (x_start, ymin + label_h - baseline//2), font,
                    font_scale, color=(10, 10, 10), thickness=font_thickness,
                    lineType=cv2.LINE_AA)

        # Draw distance lines
        cv2.line(frame, d['ground_mid'], (frame.shape[1] // 2, frame.shape[0]),
                 color=(255, 255, 255), thickness=2)

        cv2.line(birdeye, d['ground_mid_warped'],
                 (birdeye.shape[1] // 2, birdeye.shape[0]),
                 color=(255, 255, 255), thickness=3)

        # Draw ground midpoint markers
        x, y = d['ground_mid']
        cv2.circle(frame, center=(x, y), radius=5, color=color,
                   thickness=cv2.FILLED)
        cv2.circle(birdeye, center=d['ground_mid_warped'], radius=9,
                   color=color, thickness=cv2.FILLED)

        # Draw distance text
        dist_meter = d['dist_meter']
        cv2.putText(birdeye, f'{dist_meter:.02f} m', d['ground_mid_warped'],
                    font, fontScale=3, color=(255, 255, 255), thickness=8)
        x, y = d['ground_mid']
        cv2.putText(frame, f'{dist_meter:.02f} m', (x, y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, color=(255, 255, 255), thickness=3)

    return frame

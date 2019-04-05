from typing import List

import cv2
import numpy as np
import torch

from ssd.data import VOC_CLASSES as voc_labels


def ssd_preprocess(image: np.ndarray, resize_dim: int = 300):
    """
    Preprocess a BGR image loaded with opencv to be fed to SSD
    """
    x = cv2.resize(image, (resize_dim, resize_dim)).astype(np.float32)
    x -= (104.0, 117.0, 123.0)
    x = x.astype(np.float32)
    x = np.ascontiguousarray(x[:, :, ::-1])
    return torch.from_numpy(x).permute(2, 0, 1)


def postprocess_detections(ssd_detections: torch.Tensor,
                           min_confidence: float = 0.0,
                           dataset: str = 'VOC',
                           filter_classes: List=None):
    """
    Post-process SSD detections, discarding those under confidence threshold.

    :param ssd_detections: torch.Tensor of shape (batch, n_classes, n_boxes, 5) 
    :param min_confidence: Min confidence for the detection to be reliable
    :param dataset: Dataset detected classes belong to
    :param filter_classes: If present, only these classes are kept
    :return: List of post-processed detections 
    """
    if dataset != 'VOC':
        raise NotImplementedError('COCO support not implemented yet.')

    detections = ssd_detections.to('cpu').numpy()
    _, n_classes, n_detections, _ = detections.shape

    output = []

    for i in range(n_classes):

        j = 0

        # Detection proposals are already sorted by confidence score
        while detections[0, i, j, 0] >= min_confidence:
            list_item = {
                'score': detections[0, i, j, 0],
                'name': voc_labels[i - 1],
                'coords': detections[0, i, j, 1:]
            }

            j += 1

            if filter_classes is not None:
                if list_item['name'] not in filter_classes:
                    continue

            # Compute detection midpoint on the ground
            #  this is the point that will be warped by homography
            xmin, ymin, xmax, ymax = list_item['coords']
            ground_mid = xmin + (xmax - xmin) / 2, ymax
            list_item['ground_mid'] = np.asarray(ground_mid, dtype=float)

            output.append(list_item)

    return output


def draw_detections(frame: np.ndarray,
                    postprocessed_detections: List[dict],
                    color_palette: List[tuple],
                    dataset: str = 'VOC',
                    line_thickness: int = 2):

    if dataset != 'VOC':
        raise NotImplementedError('COCO support not implemented yet.')

    h, w, c = frame.shape
    for d in postprocessed_detections:
        idx = voc_labels.index(d['name'])
        color = color_palette[idx]
        color = [int(c * 255) for c in color]
        coords = [w, h, w, h] * d['coords']
        xmin, ymin, xmax, ymax = map(int, coords)
        frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color,
                              thickness=line_thickness)

        text = d['name']
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        (label_w, label_h), baseline = cv2.getTextSize(text, font,
                                                       fontScale=font_scale,
                                                       thickness=font_thickness)
        x_start = xmin - line_thickness // 2
        frame = cv2.rectangle(frame, (x_start, ymin),
                              (x_start+label_w, ymin - label_h - baseline),
                              color=color, thickness=cv2.FILLED)
        cv2.putText(frame, text, (x_start, ymin - baseline), font, font_scale,
                    color=(10, 10, 10), thickness=font_thickness,
                    lineType=cv2.LINE_AA)

        # Draw ground midpoint
        x, y = map(int, d['ground_mid'] * [w, h])
        cv2.circle(frame, center=(x, y), radius=5, color=color,
                   thickness=cv2.FILLED)

    return frame

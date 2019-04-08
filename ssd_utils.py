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


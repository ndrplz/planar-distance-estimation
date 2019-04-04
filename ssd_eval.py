from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from ssd.data import VOC_CLASSES as voc_labels
from ssd.ssd import build_ssd


def ssd_preprocess(image: np.ndarray, resize_dim: int=300):
    """
    Preprocess a BGR image loaded with opencv to be fed to SSD
    """
    x = cv2.resize(image, (resize_dim, resize_dim)).astype(np.float32)
    x -= (104.0, 117.0, 123.0)
    x = x.astype(np.float32)
    x = np.ascontiguousarray(x[:, :, ::-1])
    return torch.from_numpy(x).permute(2, 0, 1)


def postprocess_detections(ssd_detections: torch.Tensor,
                           min_confidence: float=0.0,
                           dataset:str ='VOC'):
    """
    Post-process SSD detections, discarding those under confidence threshold.
    
    :param ssd_detections: torch.Tensor of shape (batch, n_classes, n_boxes, 5) 
    :param min_confidence: Min confidence for the detection to be reliable
    :param dataset: Dataset detected classes belong to 
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

            output.append(list_item)

            j += 1

    return output


def draw_detections(frame: np.ndarray,
                    postprocessed_detections: List[dict],
                    color_palette: List[tuple],
                    dataset: str='VOC',
                    line_thickness: int=3):
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
    return frame


if __name__ == '__main__':

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    net = build_ssd('test', size=300, num_classes=21)  # initialize SSD
    net.load_weights('/media/minotauro/DATA/ai4automotive/ssd300_mAP_77.43_v2.pth')
    if torch.cuda.is_available():
        net = net.cuda()

    # todo net.eval()

    image = '/media/minotauro/DATA/ai4automotive/frames/001810.jpg'
    image = cv2.imread(image)

    x = ssd_preprocess(image, resize_dim=300)

    if torch.cuda.is_available():
        x = x.cuda()

    with torch.no_grad():
        y = net(x.unsqueeze(0))

    output = postprocess_detections(y, min_confidence=0.1)

    palette_rgb = plt.cm.hsv(np.linspace(0, 1, len(voc_labels) + 1)).tolist()
    palette_bgr = [p[:-1][::-1] for p in palette_rgb]
    # todo palette in BGR
    image_show = draw_detections(image, output, palette_bgr)

    cv2.imshow('detected', image_show)
    cv2.waitKey(0)


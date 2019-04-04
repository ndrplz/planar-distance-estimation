from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from ssd.data import VOC_CLASSES as voc_labels
from ssd.ssd import build_ssd
from ssd_utils import draw_detections
from ssd_utils import postprocess_detections
from ssd_utils import ssd_preprocess


if __name__ == '__main__':

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    net = build_ssd('test', size=300, num_classes=21)  # initialize SSD
    net.load_weights('/media/minotauro/DATA/ai4automotive/ssd300_mAP_77.43_v2.pth')
    if torch.cuda.is_available():
        net = net.cuda()

    net.eval()

    frames_dir = Path('/media/minotauro/DATA/ai4automotive/frames')
    for i, f_path in enumerate(sorted(frames_dir.glob('*.jpg'))):

        # f_path = '/media/minotauro/DATA/ai4automotive/frames/001810.jpg'
        image = cv2.imread(str(f_path))
        image = cv2.resize(image, dsize=None, fx=0.5, fy=0.5)

        x = ssd_preprocess(image, resize_dim=300)

        if torch.cuda.is_available():
            x = x.cuda()

        with torch.no_grad():
            y = net(x.unsqueeze(0))

        output = postprocess_detections(y, min_confidence=0.2,
                                        filter_classes=['car', 'person'])

        palette_rgb = plt.cm.hsv(np.linspace(0, 1, len(voc_labels) + 1)).tolist()
        palette_bgr = [p[:-1][::-1] for p in palette_rgb]

        image_show = draw_detections(image, output, palette_bgr)

        cv2.imshow('detected', image_show)
        cv2.waitKey(1)


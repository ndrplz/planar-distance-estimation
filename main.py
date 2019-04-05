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


def blend(img1, img2, alpha=0.5):
    beta = 1 - alpha
    out = np.uint8(img1 * alpha + img2 * beta)
    return out


frames_dir = Path('/media/minotauro/DATA/ai4automotive/frames')

# Trapezoid points considering full resolution (1080x1920)
h, w = 1080, 1920
y0, y1 = 550, h - 100
x_off = 50  # 280
a = 0, y1
b = w//2 - x_off, y0
c = w//2 + x_off, y0
d = w, y1

trapezoid = np.asarray([a, b, c, d])
trapezoid = np.expand_dims(trapezoid, 1).astype(np.int32)
trapezoid_img = np.zeros(shape=(h, w, 3))
trapezoid_img = cv2.fillPoly(trapezoid_img, [trapezoid], color=(0, 0, 255))

bev_w, bev_h = 1400, 2000
dst_points = np.asarray([(bev_w//2-200, bev_h),
                         (bev_w//2-200, 0),
                         (bev_w//2+200, 0),
                         (bev_w//2+200, bev_h)])
M, mask = cv2.findHomography(trapezoid, dst_points)

r = 1

# todo: class to color dictionary

pix_per_meter = 40  #

if __name__ == '__main__':

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    net = build_ssd('test', size=300, num_classes=21)  # initialize SSD
    net.load_weights('/media/minotauro/DATA/ai4automotive/ssd300_mAP_77.43_v2.pth')
    if torch.cuda.is_available():
        net = net.cuda()
    net.eval()

    for i, f_path in enumerate(sorted(frames_dir.glob('*.jpg'))[::5]):
        # if i < 2000:
        #     continue

        # f_path = '/media/minotauro/DATA/ai4automotive/frames/001810.jpg'
        frame = cv2.imread(str(f_path))
        frame = cv2.resize(frame, dsize=None, fx=r, fy=r)
        trapezoid_img = cv2.resize(trapezoid_img, dsize=None, fx=r, fy=r)

        warped = cv2.warpPerspective(frame, M, (bev_w, bev_h))

        # Preprocess frame and predict
        x = ssd_preprocess(frame, resize_dim=300)

        if torch.cuda.is_available():
            x = x.cuda()

        with torch.no_grad():
            y = net(x.unsqueeze(0))

        output = postprocess_detections(y, min_confidence=0.2,
                                        filter_classes=['car', 'person',
                                                        'bicycle'])

        # Warp midpoints in birdeye view
        for o in output:
            midpoint = o['ground_mid'] * [w, h]  # from relative to image size
            midpoint = np.concatenate([midpoint, np.ones(1)])

            midpoint_warped = M @ midpoint
            midpoint_warped /= midpoint_warped[-1]
            midpoint_warped = midpoint_warped[:-1]

            o['ground_mid_warped'] = midpoint_warped

            # Draw projected point
            x, y = map(int, midpoint_warped)

            warped = cv2.circle(warped, center=(x, y), radius=5, color=(255, 255, 255),
                       thickness=cv2.FILLED)

        # Draw distance lines
        ego_xy = w // 2, h
        ego_bev_xy = bev_w // 2, bev_h

        # Compute distances
        for o in output:
            x, y = map(int, o['ground_mid_warped'])
            delta = [x, y] - np.asarray(ego_bev_xy)
            dist_pix = np.sqrt(np.sum(delta ** 2))
            dist_meter = dist_pix / pix_per_meter

            cv2.putText(warped, f'{dist_meter:.02f} m', (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=3, color=(255, 255, 255), thickness=3)

            x, y = map(int, o['ground_mid'] * [w, h])
            cv2.putText(frame, f'{dist_meter:.02f} m', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255, 255, 255), thickness=3)

        # Draw distance lines
        for o in output:
            # Draw distance in ego view
            x, y = map(int, o['ground_mid'] * [w, h])
            cv2.line(frame, (x, y), ego_xy, color=(255, 255, 255),
                     thickness=2)

            x, y = map(int, o['ground_mid_warped'])
            cv2.line(warped, (x, y), ego_bev_xy, color=(255, 255, 255),
                     thickness=2)

        palette_rgb = plt.cm.hsv(np.linspace(0, 1, len(voc_labels) + 1)).tolist()
        palette_bgr = [p[:-1][::-1] for p in palette_rgb]

        image_show = draw_detections(frame, output, palette_bgr)

        blend_show = cv2.resize(blend(frame, trapezoid_img), dsize=None, fx=0.5, fy=0.5)
        warped_show = cv2.resize(warped, dsize=None, fx=0.5, fy=0.5)
        detect_show = cv2.resize(image_show, dsize=None, fx=0.5, fy=0.5)

        ratio = blend_show.shape[0] / warped_show.shape[0]
        warped_show = cv2.resize(warped_show, dsize=None, fx=ratio, fy=ratio)
        cat_show = np.concatenate([blend_show, warped_show], axis=1)
        cv2.imshow('warped', cat_show)
        cv2.waitKey(1)

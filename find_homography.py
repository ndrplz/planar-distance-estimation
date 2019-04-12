"""
Find homography matrix to warp from frontal to bird's eye view
"""
import cv2
import numpy as np
from view_utils import blend


if __name__ == '__main__':

    frame_frontal = cv2.imread('./data/frames/frontal.jpg')

    # Choose trapezoid points in frontal view
    height, width = frame_frontal.shape[:-1]
    y0, y1 = 550, height - 100
    x_off = 50  # 280
    a = 0, y1
    b = width // 2 - x_off, y0
    c = width // 2 + x_off, y0
    d = width, y1

    trapezoid = np.asarray([a, b, c, d])
    trapezoid = np.expand_dims(trapezoid, 1).astype(np.int32)

    trapezoid_img = cv2.fillPoly(np.zeros(shape=(height, width, 3)),
                                 [trapezoid], color=(0, 0, 255))

    # In real world, the trapezoid is a rectangle with the following measures
    ab_meters = 50
    ad_meters = 10

    # Measure of the bird's eye image
    pix_per_meter = 40
    bev_w = ad_meters * pix_per_meter
    bev_h = ab_meters * pix_per_meter
    xoff = 500
    dst_points = np.asarray([(xoff + 0, bev_h),
                             (xoff + 0, 0),
                             (xoff + bev_w, 0),
                             (xoff + bev_w, bev_h)])
    bev_w += (2 * xoff)

    H, mask = cv2.findHomography(trapezoid, dst_points)
    np.savez('./data/homography.npz', H=H, bev_h=bev_h, bev_w=bev_w,
             pix_per_meter=pix_per_meter)

    # Warp according to currently found homography
    birdeye = cv2.warpPerspective(frame_frontal, H, (bev_w, bev_h))

    image_show = blend(trapezoid_img, frame_frontal)

    ratio = image_show.shape[0] / birdeye.shape[0]
    warped_show = cv2.resize(birdeye, dsize=None, fx=ratio, fy=ratio)
    cat_show = np.concatenate([image_show, warped_show], axis=1)
    cat_show = cv2.resize(cat_show, dsize=None, fx=0.5, fy=0.5)
    cv2.imshow('blend', cat_show)
    cv2.waitKey(0)





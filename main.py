"""
This script estimate the distance of visible on-road objects, given:
    * pre-computed homography matrix between to warp road plane
    * pre-trained YOLOv3 object detector (args.yolo_weights)
"""
from __future__ import division

import argparse
from pathlib import Path

import cv2
from torch.utils.data import DataLoader

from view_utils import draw_detections
from view_utils import name_to_color
from yolo.models import *
from yolo.utils.datasets import *
from yolo.utils.utils import *
from yolo_utils import postprocess_yolo_detections


parser = argparse.ArgumentParser()
parser.add_argument('yolo_weights', type=str, help='Pre-trained YOLO weights')
parser.add_argument('--frames_dir', default=Path('./data/frames'))
parser.add_argument('--homography_data', default=Path('./data/homography.npz'),
                    help='Pre-computed homography')
parser.add_argument('--output_dir', type=Path, default='/tmp/distances')
parser.add_argument('--config_path', type=str, default='config/yolov3.cfg', help='path to model config file')
parser.add_argument('--class_path', type=str, default='yolo/data/coco.names', help='path to class label file')
parser.add_argument('--conf_thres', type=float, default=0.7, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.4, help='IoI threshold for non-maximum suppression')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--n_cpu', type=int, default=8, help='data loader threads')
parser.add_argument('--img_size', type=int, default=416, help='size of each image dimension')
parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
args = parser.parse_args()
print(args)


if __name__ == '__main__':

    # Load pre-computed homography matrix
    h_data = np.load(args.homography_data)
    H = h_data['H']
    bev_h, bev_w = h_data['bev_h'], h_data['bev_w']
    pix_per_meter = h_data['pix_per_meter']

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # Set up detector
    model = Darknet(config_path='yolo/config/yolov3.cfg', img_size=args.img_size)
    model.load_weights(args.yolo_weights)
    if args.device == 'cuda':
        model.cuda()

    Tensor = torch.cuda.FloatTensor if args.device == 'cuda' else torch.FloatTensor

    dataloader = DataLoader(
        ImageFolder(args.frames_dir, img_size=args.img_size),
        batch_size=1, shuffle=False, num_workers=args.n_cpu)

    for batch_i, (img_path, input_img) in enumerate(dataloader):

        frame = cv2.imread(img_path[0])

        birdeye = cv2.warpPerspective(frame, H, (bev_w, bev_h))

        # Get detections
        with torch.no_grad():
            input_img = Variable(input_img.type(Tensor))
            detections = model(input_img)
            detections = non_max_suppression(detections, 80, args.conf_thres,
                                             args.nms_thres)
        if detections[0] is not None:
            keep_classes = ['car', 'person', 'bicycle', 'truck',
                            'bus', 'motorbike']
            output = postprocess_yolo_detections(detections[0],
                                                 image_shape=frame.shape,
                                                 input_size=args.img_size,
                                                 filter_classes=keep_classes)

            # Warp midpoints in birdeye view
            for o in output:
                midpoint = np.concatenate([o['ground_mid'], np.ones(1)])

                midpoint_warped = H @ midpoint
                midpoint_warped /= midpoint_warped[-1]
                midpoint_warped = midpoint_warped[:-1]

                # Store projected points
                o['ground_mid_warped'] = tuple(int(a) for a in midpoint_warped)

            # Exploit the birdeye view to compute distances
            ego_bev_xy = bev_w // 2, bev_h  # Ego position in birdeye view
            for o in output:
                x, y = o['ground_mid_warped']
                delta = [x, y] - np.asarray(ego_bev_xy)
                dist_pix = np.sqrt(np.sum(delta ** 2))

                o['dist_meter'] = dist_pix / pix_per_meter

            # Draw bounding boxes, text, lines etc.
            frame = draw_detections(frame, birdeye, output, name_to_color)

        resize_f = 0.6
        warped_show = cv2.resize(birdeye, dsize=None, fx=resize_f, fy=resize_f)
        image_show = cv2.resize(frame, dsize=None, fx=resize_f, fy=resize_f)
        blend_show = image_show

        ratio = blend_show.shape[0] / warped_show.shape[0]
        warped_show = cv2.resize(warped_show, dsize=None, fx=ratio, fy=ratio)
        cat_show = np.concatenate([blend_show, warped_show], axis=1)

        if not args.output_dir.exists():
            args.output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.output_dir / f'{batch_i:06d}.jpg'), cat_show)

        # Show output
        cv2.imshow('Output (press any key to proceed)', cat_show)
        cv2.waitKey(0)

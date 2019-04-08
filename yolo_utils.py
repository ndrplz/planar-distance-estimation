import torch

from yolo.utils.utils import load_classes


# Extracts class labels from file
coco_classes = load_classes('yolo/data/coco.names')


def postprocess_yolo_detections(yolo_detections: torch.Tensor,
                                input_size: int, image_shape: int,
                                filter_classes=None):
    """
    Post-process YOLO detections 
    """

    # The amount of padding that was added
    h, w, c = image_shape
    pad_x = max(h - w, 0) * (input_size / max([h, w]))
    pad_y = max(w - h, 0) * (input_size / max([h, w]))
    # Image height and width after padding is removed
    unpad_h = input_size - pad_y
    unpad_w = input_size - pad_x

    output = []

    detections = yolo_detections.to('cpu').numpy()
    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
        # Rescale coordinates to original dimensions
        box_h = ((y2 - y1) / unpad_h) * h
        box_w = ((x2 - x1) / unpad_w) * w
        y1 = ((y1 - pad_y // 2) / unpad_h) * h
        x1 = ((x1 - pad_x // 2) / unpad_w) * w

        list_item = {
            'name': coco_classes[int(cls_pred)],
            'coords': [int(a) for a in (x1, y1, x1+box_w, y1+box_h)],
            'score': conf,
            'score_cls': cls_conf,
            'class_idx': int(cls_pred),
            # Compute detection midpoint on the ground
            #  this is the point that will be warped by homography
            'ground_mid': (int(x1 + box_w // 2), int(y1 + box_h))
        }

        if filter_classes is not None:
            if list_item['name'] not in filter_classes:
                continue

        output.append(list_item)

    return output

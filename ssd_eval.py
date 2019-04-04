import cv2
import numpy as np
import torch
from ssd.data import VOC_CLASSES as labels
from ssd.ssd import build_ssd
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


def ssd_preprocess(image: np.ndarray, resize_dim: int=300):
    """
    Preprocess a BGR image loaded with opencv to be fed to SSD
    """
    x = cv2.resize(image, (resize_dim, resize_dim)).astype(np.float32)
    x -= (104.0, 117.0, 123.0)
    x = x.astype(np.float32)
    x = np.ascontiguousarray(x[:, :, ::-1])
    return torch.from_numpy(x).permute(2, 0, 1)


net = build_ssd('test', size=300, num_classes=21)  # initialize SSD
net.load_weights('/media/minotauro/DATA/ai4automotive/ssd300_mAP_77.43_v2.pth')
if torch.cuda.is_available():
    net = net.cuda()

image = '/media/minotauro/DATA/ai4automotive/frames/001810.jpg'
image = cv2.imread(image)

x = ssd_preprocess(image, resize_dim=300)

if torch.cuda.is_available():
    x = x.cuda()

with torch.no_grad():
    y = net(x.unsqueeze(0))

min_confidence = 0.1

plt.figure(figsize=(10, 10))
colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
plt.imshow(image[:, :, ::-1])  # plot the image for matplotlib
currentAxis = plt.gca()

detections = y.data
# scale each detection back up to the image
scale = torch.Tensor(image.shape[1::-1]).repeat(2)
for i in range(detections.size(1)):
    j = 0
    while detections[0, i, j, 0] >= min_confidence:
        score = detections[0, i, j, 0]
        label_name = labels[i - 1]
        display_txt = '%s: %.2f' % (label_name, score)
        pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
        coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
        color = colors[i]
        currentAxis.add_patch(
            plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(pt[0], pt[1], display_txt,
                         bbox={'facecolor': color, 'alpha': 0.5})
        j += 1
plt.show()

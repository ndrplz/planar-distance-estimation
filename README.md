# ai4automotive

This repository accompanies the laboratory practice on **Planar Distance Estimation** for the *AI4Automotive* course at [University of Modena and Reggio Emilia](http://www.international.unimore.it/). Please note that this implementation has only a didactic purpose. 

## Run the code

The code entry point is `main.py`.

Please notice that to reproduce results in the demo the following things are needed:
* Videos from the [DR(eye)VE dataset](http://aimagelab.ing.unimore.it/dreyeve). The demo is run on sequence 06. The code expects to find unrolled frames in `data/frames`. One example frame is already in `data`.
* Homography matrix to warp the road plane from frontal to bird's eye view. Example matrix is included in `data`.
* YOLOv3 detector pre-trained weights (downloadable [here](https://pjreddie.com/media/files/yolov3.weights)).

The PyTorch wrapper to YOLOv3 detector is borrowed from [this](https://github.com/eriklindernoren/PyTorch-YOLOv3) clean implementation by [eriklindernoren](https://github.com/eriklindernoren). Thanks Erik! :)

## Slides
<p align="center">
  <img src="data/slide_thumb.png"/ alt="Slides thumbnail" width="25%">
  <br>
  <br>Slides of the practice can be downloaded <a href="https://drive.google.com/open?id=1J4K731Lu55oCZJVDuA6DzG54dzZRlV-J" name="p4_code">here</a>.
</p>


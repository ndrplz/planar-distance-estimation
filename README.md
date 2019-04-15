# ai4automotive

This repository accompanies the laboratory practice on **Planar Distance Estimation** for the *AI4Automotive* course at [University of Modena and Reggio Emilia](http://www.international.unimore.it/). Please note that this implementation has only a didactic purpose. 

<p align="center">
  <a href="https://www.youtube.com/watch?v=Fk32Pz3LzCQ">
  <img src="data/teaser.gif"/ alt="Qualitative results" width="75%">
  </a>
  <br>Qualitative results. Whole video <a href="https://www.youtube.com/watch?v=Fk32Pz3LzCQ" name="video">here</a>.
</p>

## Run the code

The code entry point is `main.py`.

Please notice that to reproduce results in the demo the following things are needed:
* Videos from the [DR(eye)VE dataset](http://aimagelab.ing.unimore.it/dreyeve). The demo is run on sequence 06. The code expects to find unrolled frames in `data/frames`. One example frame is already in `data`.
* Homography matrix to warp the road plane from frontal to bird's eye view. Example matrix is included in `data`.
* YOLOv3 detector pre-trained weights (downloadable [here](https://pjreddie.com/media/files/yolov3.weights)).

The PyTorch wrapper to YOLOv3 detector is borrowed from [this](https://github.com/eriklindernoren/PyTorch-YOLOv3) clean implementation by [eriklindernoren](https://github.com/eriklindernoren). Thanks Erik! :)

## Slides
<p align="center">
  <a href="https://drive.google.com/open?id=1J4K731Lu55oCZJVDuA6DzG54dzZRlV-J">
  <img src="data/slide_thumb.png"/ alt="Slides thumbnail" width="40%">
  </a>
  <br>
  <br>Slides of the practice are available <a href="https://drive.google.com/open?id=1J4K731Lu55oCZJVDuA6DzG54dzZRlV-J" name="slides">here</a>.
</p>


# ObjectTrajectories
Estimate object trajectories in an automotive context, using KITTI dataset, with CNNs.

A very detailed description of the implamentation is available in the `relazione.pdf` file (but in italian).

[![TestImage1](https://s14.postimg.org/61x9041k1/consistenza.png)](https://postimg.org/image/5p5utxja5/)

[![TestImage2](https://s14.postimg.org/u6xyhtlup/occlusioni.png)](https://postimg.org/image/6fykzplnh/)

The project dependencies are the following Python packets (modules):
- Pillow
- Numpy
- GeoPy
- Keras
- TensorFlow


The CNNs code is based on the following GitHub repos:

- [YOLO: YAD2K - Yet Another Darknet 2 Keras](https://github.com/allanzelener/YAD2K)
- [SSD: Single Shot MultiBox Detector in Keras](https://github.com/oarriaga/single_shot_multibox_detector)

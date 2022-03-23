# Few Shot Learning

Implementation of a few shot learning algorithm using Tensorflow.js with a mobilenetv2 backbone and a Knn-Classifier on top of it.

This project is based on the official tutorial: https://www.tensorflow.org/js/tutorials/transfer/image_classification

## Usage

1. To classify image from disk: Run classifier/index.html
2. To classify image from webcam: Run classifier_camera.html

## Demo

1. To classify images from disk use [this](https://martingramatica.com/classifier/) link
2. To classify image from webcam use [this](https://martingramatica.com/classifier_camera/) link

## Data
Example data is included wit 3 datasets

- Animals
    - Cats: 10 images
    - Dogs: 10 images
    - Lions: 10 images
- Animals_inference
    - Cats: 1 image
    - Dogs: 1 image
    - Lions: 1 image
    - Fake lions: 2 images
- Blur_detection
    - Blured: 5 images
    - Focused: 5 items
- Colors
    - Blue: 10 images
    - Green: 10 images
    - Red: 10 images
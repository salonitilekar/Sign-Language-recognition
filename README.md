## Sign-language-recognition

This repository contains python implementation for recognising sign language (ISL) gestures. There is no standard dataset avialable in the web. So, I used one readily available ISL dataset of gesture images. ISL dataset have all alphabets (A-Z) and numerics (1-9) with total classes = 35. Each class has 1200 images. ISL gestures are practically hard to recognise as two hands are involved and because of complexity. To classify images, Bag-of-words (bow) model has been implemented with SVM.

## Required setup
* python 3 or above
* opencv-python
* opencv-contrib-python
* numpy
* imutils

## Implementation

The implementation follows several steps:

i) Image segmentation (masking to get raw skin and edges in the image) <br/>
ii) SIFT Feature detection (finding feature descriptors for all data) <br/>
iii) Minibatch K-means clustering <br/>
iv) Histograms computation (Using visual words (bow) compute histograms for each image) <br/>
v) SVM model for classification (input: histograms, output: predection for testdata) <br/>

## Results
As the dataset images are much similar, the model is giving fair accuracy which can be improved futher

## Run files

Run files in order:<br/>

**Step 1:** extract your dataset in the root directory of the repository.  Then run

>   python framePreprocessing.py

to preprocess all the images (from raw images to histograms of bovw model) and to classify using SVM.

**Step 2:** To recognise real-time gesture

>   python identifyGesture.py

Dataset can be downloaded from : https://drive.google.com/file/d/1dSVNl35erbOTfAMpwsRRoOuXr8JK70Cd/view?usp=drive_link

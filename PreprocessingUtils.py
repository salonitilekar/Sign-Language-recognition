import numpy as np
import cv2
import os
import random
import pickle
from imutils import paths

# Configurable Parameters
PATH = 'data'
TRAIN_FACTOR = 80
TOTAL_IMAGES = 1200
N_CLASSES = 35
CLUSTER_FACTOR = 8

# ROI coordinates
START = (400, 75)
END = (600, 300)
IMG_SIZE = 128

def get_canny_edge(image):
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    HSVImaage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lowerBoundary = np.array([0, 40, 30], dtype="uint8")
    upperBoundary = np.array([43, 255, 254], dtype="uint8")
    skinMask = cv2.inRange(HSVImaage, lowerBoundary, upperBoundary)
    skinMask = cv2.addWeighted(skinMask, 0.5, skinMask, 0.5, 0.0)
    skinMask = cv2.medianBlur(skinMask, 5)
    skin = cv2.bitwise_and(grayImage, grayImage, mask=skinMask)
    canny = cv2.Canny(skin, 50, 100)  # Updated thresholds
    return canny, skin

def get_SIFT_descriptors(image):
    sift = cv2.SIFT_create()
    image = cv2.resize(image, (256, 256))  # Standard resizing
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return descriptors

def get_labels():
    class_labels = []
    for (dirpath, dirnames, filenames) in os.walk(PATH):
        dirnames.sort()
        for label in dirnames:
            if not (label == '.DS_Store'):
                class_labels.append(label)
    return class_labels

def get_all_gestures():
    gestures = []
    for (dirpath, dirnames, filenames) in os.walk(PATH):
        dirnames.sort()
        for label in dirnames:
            if not (label == '.DS_Store'):
                for (subdirpath, subdirnames, images) in os.walk(PATH + '/' + label + '/'):
                    random.shuffle(images)
                    imagePath = PATH + '/' + label + '/' + images[0]
                    img = cv2.imread(imagePath)
                    img = cv2.resize(img, (int(IMG_SIZE * 3 / 4), int(IMG_SIZE * 3 / 4)))
                    img = cv2.putText(img, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    gestures.append(img)

    print('Length of gestures: {}'.format(len(gestures)))
    im_tile = concat_tile(gestures, (5, 7))
    return im_tile

def concat_tile(im_list_2d, size):
    count = 0
    all_imgs = []
    for row in range(size[1]):
        imgs = []
        for col in range(size[0]):
            imgs.append(im_list_2d[count])
            count += 1
        all_imgs.append(imgs)
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in all_imgs])
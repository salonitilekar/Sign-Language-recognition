import numpy as np
import cv2
import os
import pickle
import imagePreprocessingUtils as ipu

CAPTURE_FLAG = False

class_labels = ipu.get_labels()

def recognise(cluster_model, classify_model):
    global CAPTURE_FLAG
    gestures = ipu.get_all_gestures()
    cv2.imwrite("all_gestures.jpg", gestures)
    camera = cv2.VideoCapture(0)
    print('Camera window will open.\n1) Place your hand gesture in the ROI (rectangle).\n2) Press "p" to capture the gesture.\n3) Press "Esc" to exit.')

    while True:
        ret, frame = camera.read()
        frame = cv2.flip(frame, 1)
        cv2.rectangle(frame, ipu.START, ipu.END, (0, 255, 0), 2)
        cv2.imshow("All_gestures", gestures)
        pressedKey = cv2.waitKey(1)

        if pressedKey == 27:  # ESC key to exit
            break
        elif pressedKey == ord('p'):  # 'p' key to capture
            CAPTURE_FLAG = not CAPTURE_FLAG

        if CAPTURE_FLAG:
            roi = frame[ipu.START[1]+5:ipu.END[1], ipu.START[0]+5:ipu.END[0]]
            if roi is not None:
                roi = cv2.resize(roi, (ipu.IMG_SIZE, ipu.IMG_SIZE))
                canny_edge = ipu.get_canny_edge(roi)[0]
                sift_descriptors = ipu.get_SIFT_descriptors(canny_edge)

            if sift_descriptors is not None:
                visual_words = cluster_model.predict(sift_descriptors)
                bovw_histogram = np.array(np.bincount(visual_words, minlength=ipu.N_CLASSES * ipu.CLUSTER_FACTOR))
                prediction = classify_model.predict([bovw_histogram])
                label = class_labels[prediction[0]]

                cv2.putText(frame, f"Predicted: {label}", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Video", frame)

    camera.release()
    cv2.destroyAllWindows()

clustering_model = pickle.load(open('mini_kmeans_model.sav', 'rb'))
classification_model = pickle.load(open('svm_model.sav', 'rb'))
recognise(clustering_model, classification_model)
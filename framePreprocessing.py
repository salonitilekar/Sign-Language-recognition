import numpy as np
import cv2
import os
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import random
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import PreprocessingUtils as ipu

# Labels
train_labels = []
test_labels = []

# Preprocessing function
def preprocess_all_images():
    train_img_disc = []
    test_img_disc = []
    all_train_dis = []
    label_value = 0

    for dirpath, dirnames, filenames in os.walk(ipu.PATH):
        dirnames.sort()
        for label in dirnames:
            if label != ".DS_Store":
                for subdirpath, subdirnames, images in os.walk(f"{ipu.PATH}/{label}/"):
                    count = 0
                    for image in images:
                        image_path = f"{ipu.PATH}/{label}/{image}"
                        img = cv2.imread(image_path)
                        if img is not None:
                            img = preprocess_image(img)
                            sift_desc = get_SIFT_descriptors(img)

                            if sift_desc is not None:
                                if count < (ipu.TOTAL_IMAGES * ipu.TRAIN_FACTOR * 0.01):
                                    train_img_disc.append(sift_desc)
                                    all_train_dis.extend(sift_desc)
                                    train_labels.append(label_value)
                                elif count < ipu.TOTAL_IMAGES:
                                    test_img_disc.append(sift_desc)
                                    test_labels.append(label_value)
                                count += 1
                    label_value += 1

    print(f"Length of train descriptors: {len(train_img_disc)}")
    print(f"Length of test descriptors: {len(test_img_disc)}")
    print(f"Length of all train descriptors: {len(all_train_dis)}")

    return all_train_dis, train_img_disc, test_img_disc

# Image preprocessing
def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    return cv2.Canny(blurred, 50, 150)

# SIFT descriptor extraction
def get_SIFT_descriptors(img, max_keypoints=500):
    sift = cv2.SIFT_create(nfeatures=max_keypoints)
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return descriptors

# MiniBatch KMeans clustering
def mini_kmeans(k, descriptor_list):
    print("MiniBatch KMeans started.")
    kmeans_model = MiniBatchKMeans(k)
    kmeans_model.fit(descriptor_list)
    pickle.dump(kmeans_model, open("mini_kmeans_model.sav", "wb"))
    return kmeans_model

# Normalize histogram
def normalize_histogram(hist):
    return hist / np.sum(hist) if np.sum(hist) > 0 else hist

# Plot histogram
def plot_histogram(hist, title):
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(hist)), hist)
    plt.title(title)
    plt.show()

# SVM training with GridSearchCV
def tune_svm(X_train, y_train):
    svc = SVC()
    params = {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto']
    }
    grid_search = GridSearchCV(svc, param_grid=params, cv=5)
    grid_search.fit(X_train, y_train)
    print("Best Parameters:", grid_search.best_params_)
    return grid_search.best_estimator_

# Evaluation metrics
def calculate_metrics(method, y_test, y_pred):
    print(f"Accuracy score for {method}: {accuracy_score(y_test, y_pred)}")
    print(f"Precision score for {method}: {precision_score(y_test, y_pred, average='micro')}")
    print(f"Recall score for {method}: {recall_score(y_test, y_pred, average='micro')}")
    print(f"F1 score for {method}: {f1_score(y_test, y_pred, average='micro')}")

# Confusion matrix
def plot_confusion_matrix(y_test, y_pred, class_names):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# Main process
if __name__ == "__main__":
    # Step 1: Preprocess images
    all_train_dis, train_img_disc, test_img_disc = preprocess_all_images()

    # Step 2: Train MiniBatchKMeans
    cluster_count = ipu.N_CLASSES * ipu.CLUSTER_FACTOR
    mini_kmeans_model = mini_kmeans(cluster_count, np.array(all_train_dis))

    # Step 3: Create BoVW histograms
    train_visual_words = [mini_kmeans_model.predict(desc) for desc in train_img_disc]
    test_visual_words = [mini_kmeans_model.predict(desc) for desc in test_img_disc]

    bovw_train_histograms = np.array([normalize_histogram(np.bincount(vw, minlength=cluster_count)) for vw in train_visual_words])
    bovw_test_histograms = np.array([normalize_histogram(np.bincount(vw, minlength=cluster_count)) for vw in test_visual_words])

    print(f"Each histogram length: {len(bovw_train_histograms[0])}")

    # Step 4: Train SVM
    X_train, Y_train = list(bovw_train_histograms), list(train_labels)
    X_test, Y_test = list(bovw_test_histograms), list(test_labels)

    svc = tune_svm(X_train, Y_train)
    y_pred = svc.predict(X_test)

    # Step 5: Evaluate results
    calculate_metrics("SVM", Y_test, y_pred)
    plot_confusion_matrix(Y_test, y_pred, class_names=sorted(set(train_labels)))

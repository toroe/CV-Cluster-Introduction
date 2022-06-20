import numpy as np
import cv2
import glob
from sklearn import svm
import random


############################################################
#
#              Support Vector Machine
#              Image Classification
#
############################################################
def create_keypoints(w, h, w_step, h_step, kp_size):
    keypoints = []
    keypointSize = kp_size
    
    # YOUR CODE HERE
    for width in range(0, w, w_step):
        for height in range(0, h, h_step):
            keypoints.append(cv2.KeyPoint(width, height, keypointSize))

    return keypoints

# 1. Implement a SIFT feature extraction for a set of training images ./images/db/train/** (see 2.3 image retrieval)
# use ~15x15 keypoints on each image with subwindow of 21px (diameter)
test_images = glob.glob("images/db/test/*")
train_images = glob.glob("images/db/train/*/*")

descriptors = []
keypoints = create_keypoints(256, 256, int(256/15), int(256/15), 21)

y_train = np.zeros((len(train_images)))
sift = cv2.SIFT_create()
for idx, train_img in enumerate(train_images):
    img = cv2.imread(train_img, 0)
    kp, des = sift.compute(img, keypoints)
    if "cars" in train_img:
        y_train[idx] = 0
    if "faces" in train_img:
         y_train[idx] = 1
    if "flowers" in train_img:
         y_train[idx] = 2
    descriptors.append(des)

# 2. each descriptor (set of features) need to be flattened in one vector
# That means you need a X_train matrix containing a shape of (num_train_images, num_keypoints*num_entry_per_keypoint)
# num_entry_per_keypoint = histogram orientations as talked about in class
# You also need a y_train vector containing the labels encoded as integers
X_train = np.zeros((len(train_images), descriptors[0].flatten().shape[0]))
for idx, des in enumerate(descriptors):
    X_train[idx] = des.flatten()
# 3. We use scikit-learn to train a SVM classifier - however you need to test with different kernel options to get
# good results for our dataset.
debug = 0
clf = svm.SVC()
clf.fit(X_train, y_train)
# 4. We test on a variety of test images ./images/db/test/ by extracting an image descriptor
# the same way we did for the training (except for a single image now) and use .predict()
# to classify the image
for pred_img in test_images:
    img = cv2.imread(pred_img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.compute(gray, keypoints)
    output = clf.predict(des.flatten().reshape(1,-1))
    if "car" in pred_img:
        print("Actual class: " + str(0))
    if "face" in pred_img:
        print("Actual class: " + str(1))
    if "flower" in pred_img:
        print("Actual class: " + str(2))
    print("Predicted class: " + str(output[0]))
# 5. output the class + corresponding name


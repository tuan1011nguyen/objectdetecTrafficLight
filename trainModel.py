from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout
from keras.applications import VGG16
import numpy as np
from tensorflow.keras.optimizers import Adam

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.utils import to_categorical
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D
import pickle

print("[INFO] loading dataset...")
data = []
labels = []
bboxes = []
imagePaths = []

path = r"./Traffic-Lights-1/train/_annotations.csv"

if not os.path.isfile(path):
    print(f"[ERROR] CSV file not found: {path}")
else:
    with open(path) as f:
        rows = f.read().strip().split("\n")

    for row in rows[1:]:  
        row = row.split(",")
        filename, _, _, label, xmin, ymin, xmax, ymax = row
        imagePath = os.path.sep.join([r"./Traffic-Lights-1/train", filename])

        if not os.path.isfile(imagePath):
            print(f"[WARNING] Image not found: {imagePath}")
            continue
        
        image = load_img(imagePath, target_size=(64, 64))
        image = img_to_array(image)

        image_cv = cv2.imread(imagePath)
        if image_cv is not None:
            (h, w) = image_cv.shape[:2]
            startX = float(xmin) / w
            startY = float(ymin) / h
            endX = float(xmax) / w
            endY = float(ymax) / h

            data.append(image)
            labels.append(label)
            bboxes.append((startX, startY, endX, endY))
            imagePaths.append(imagePath)
        else:
            print(f"[ERROR] Unable to load image: {imagePath}")

data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)
bboxes = np.array(bboxes, dtype="float32")
imagePaths = np.array(imagePaths)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

if len(lb.classes_) == 2:
    labels = to_categorical(labels)

trainPaths, tempPaths, trainLabels, tempLabels, trainBBoxes, tempBBoxes = train_test_split(
    imagePaths, labels, bboxes, test_size=0.20, random_state=42
)

valPaths, testPaths, valLabels, testLabels, valBBoxes, testBBoxes = train_test_split(
    tempPaths, tempLabels, tempBBoxes, test_size=0.50, random_state=42
)

trainImages = np.array([img_to_array(load_img(path, target_size=(64, 64))) for path in trainPaths]) / 255.0
valImages = np.array([img_to_array(load_img(path, target_size=(64, 64))) for path in valPaths]) / 255.0
testImages = np.array([img_to_array(load_img(path, target_size=(64, 64))) for path in testPaths]) / 255.0


base_model = VGG16(weights = "imagenet", include_top = False, input_tensor = Input(shape=(64, 64, 3)))

base_model.trainable = False

flatten = base_model.output
flatten = Flatten()(flatten)

bboxHead = Dense(128, activation="relu")(flatten)
bboxHead = Dense(64, activation="relu")(bboxHead)
bboxHead = Dense(32, activation="relu")(bboxHead)
bboxHead = Dense(4, activation="sigmoid", name="bounding_box")(bboxHead)

softmaxHead = Dense(512, activation="relu")(flatten)
softmaxHead = Dropout(0.5)(softmaxHead)
softmaxHead = Dense(512, activation="relu")(softmaxHead)
softmaxHead = Dropout(0.5)(softmaxHead)
softmaxHead = Dense(len(lb.classes_), activation="softmax", name="class_label")(softmaxHead)

model = Model(inputs=base_model.input,outputs=(bboxHead,softmaxHead))

losses = {"class_label": "categorical_crossentropy", "bounding_box":"mean_squared_error"}

lossWeights = {"class_label":1.0, "bounding_box":1.0}

opt = Adam(learning_rate=0.0001)
model.compile(loss = losses, optimizer= opt, metrics = ["accuracy", "mean_squared_error"], loss_weights=lossWeights)
print(model.summary())


trainTargets = {"class_label": trainLabels, "bounding_box": trainBBoxes}

testTargets = {"class_label": testLabels, "bounding_box": testBBoxes}

print("[INFO] trainig model...")
H = model.fit(trainImages, trainTargets, validation_data = (testImages, testTargets), batch_size = 8, epochs= 50, verbose=1)

print("[INFO] saving object detector model...")
model.save("./model.h5")
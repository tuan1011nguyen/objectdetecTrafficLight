from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import pickle
import cv2
import os

image_dir = r"./Traffic-Lights-1/valid"
imagePaths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

print("[INFO] loading object detector...")
model = load_model(r"./model.h5")

lb = pickle.loads(open("label_binarizer.pickle", "rb").read())

for imagePath in imagePaths:
    image = load_img(imagePath, target_size=(64, 64))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    (boxPreds, labelPreds) = model.predict(image)
    (startX, startY, endX, endY) = boxPreds[0]

    i = np.argmax(labelPreds, axis=1)
    label = str(lb.classes_[i][0])  

    original_image = cv2.imread(imagePath)
    original_image = imutils.resize(original_image, width=600)
    (h, w) = original_image.shape[:2]

    startX = int(startX * w)
    startY = int(startY * h)
    endX = int(endX * w)
    endY = int(endY * h)

    startX = max(0, startX)
    startY = max(0, startY)
    endX = min(w, endX)
    endY = min(h, endY)

    y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.putText(original_image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.rectangle(original_image, (startX, startY), (endX, endY), (0, 255, 0), 2)

    cv2.imshow("Output", original_image)
    cv2.waitKey(0)


cv2.destroyAllWindows()

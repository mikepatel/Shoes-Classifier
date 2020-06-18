"""
Michael Patel
June 2020

Project description:
    Binary shoe classifier between Nike and Adidas

File description:
    For model inference
"""
################################################################################
# Imports
import os
import numpy as np
import cv2

import tensorflow as tf


################################################################################
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_CHANNELS = 3

TRAIN_DIR = os.path.join(os.getcwd(), "data\\train")
SAVE_DIR = os.path.join(os.getcwd(), "saved_model")


################################################################################
# Main
if __name__ == "__main__":
    # labels
    classes = []
    int2class = {}
    directories = os.listdir(TRAIN_DIR)
    for i in range(len(directories)):
        name = directories[i]
        classes.append(name)
        int2class[i] = name

    num_classes = len(classes)

    # load model
    model = tf.keras.models.load_model(SAVE_DIR)

    # webcam
    capture = cv2.VideoCapture(0)
    while True:
        # capture frames
        ret, frame = capture.read()

        # preprocess image
        image = frame
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # crop webcam

        # resize
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

        # modified image
        mod_image = image

        # to array
        image = np.array(image).astype(np.float32) / 255.0
        image = np.expand_dims(image, 0)

        # make prediction
        prediction = model.predict(image)
        predict_name = int2class[int(np.argmax(prediction))]
        print(f'{predict_name}')

        # display frame or modified image
        cv2.imshow("", frame)

        # continuous stream, break with ESC key
        if cv2.waitKey(1) == 27:
            break

    # release capture
    capture.release()
    cv2.destroyAllWindows()

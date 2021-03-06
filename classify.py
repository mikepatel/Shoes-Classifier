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
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import tensorflow as tf

from parameters import *


################################################################################
# Main
if __name__ == "__main__":
    # labels
    classes = []
    int2class = {}
    directories = os.listdir(TRAIN_DIR)
    for i in range(len(directories)):
        name = str(directories[i]).title()
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
        original_image = frame
        image = frame
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # crop webcam
        y, x, channels = image.shape
        left_x = int(x*0.25)
        right_x = int(x*0.75)
        top_y = int(y*0.25)
        bottom_y = int(y*0.75)
        image = image[top_y:bottom_y, left_x:right_x]

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

        # save image with prediction text
        pred_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        pred_image = Image.fromarray(pred_image)
        draw = ImageDraw.Draw(pred_image)
        font = ImageFont.truetype("arial.ttf", 40)
        draw.text((0, 0), predict_name, font=font)
        pred_image.save(os.path.join(os.getcwd(), "prediction2.png"))

        # display frame or modified image
        cv2.imshow("", mod_image)

        # continuous stream, break with ESC key
        if cv2.waitKey(1) == 27:
            break

    # release capture
    capture.release()
    cv2.destroyAllWindows()

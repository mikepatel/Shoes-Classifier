"""


"""
################################################################################
# Imports
import os
import numpy as np

import tensorflow as tf


################################################################################
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
NUM_CHANNELS = 3

NUM_EPOCHS = 1
BATCH_SIZE = 16


################################################################################
# Main
if __name__ == "__main__":
    # labels
    classes = []
    int2class = {}
    train_dir = os.path.join(os.getcwd(), "data\\train")
    test_dir = os.path.join(os.getcwd(), "data\\test")
    directories = os.listdir(train_dir)
    for i in range(len(directories)):
        name = directories[i]
        classes.append(name)
        int2class[i] = name

    num_classes = len(classes)
    print(f'{classes}')
    print(f'Number of classes: {num_classes}')

    # image data generator
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255  # [0, 255] --> [0, 1]
    )

    train_data_gen = image_generator.flow_from_directory(
        directory=train_dir,
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        color_mode="rgb",
        class_mode="binary",
        classes=classes,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

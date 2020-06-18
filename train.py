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
IMAGE_CHANNELS = 3

NUM_EPOCHS = 1
BATCH_SIZE = 16

TRAIN_DIR = os.path.join(os.getcwd(), "data\\train")
TEST_DIR = os.path.join(os.getcwd(), "data\\test")
TEMP_DIR = os.path.join(os.getcwd(), "data\\temp")


################################################################################
# Main
if __name__ == "__main__":
    # ----- ETL ----- #
    # ETL = Extraction, Transformation, Load
    # labels
    classes = []
    int2class = {}
    directories = os.listdir(TRAIN_DIR)
    for i in range(len(directories)):
        name = directories[i]
        classes.append(name)
        int2class[i] = name

    num_classes = len(classes)
    #print(f'{classes}')
    #print(f'Number of classes: {num_classes}')

    # image data generator
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255  # [0, 255] --> [0, 1]
    )

    train_data_gen = image_generator.flow_from_directory(
        directory=TRAIN_DIR,
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        color_mode="rgb",
        class_mode="binary",
        classes=classes,
        batch_size=BATCH_SIZE,
        shuffle=True
        #save_to_dir=TEMP_DIR
    )

    #next(train_data_gen)

    # ----- MODEL ----- #
    # will use VGG16 model with some modifications
    vgg16 = tf.keras.applications.vgg16.VGG16(
        input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS),
        include_top=False  # removed last 3 dense layers, as VGG16 is for 224x224x3
    )

    # freeze early VGG16 layers
    for layer in vgg16.layers:
        layer.trainable = False

    # build model
    model = tf.keras.models.Sequential()
    model.add(vgg16)

    # add custom output layers
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(
        units=1024,
        activation=tf.keras.activations.relu
    ))
    model.add(tf.keras.layers.Dense(
        units=num_classes,
        activation=tf.keras.activations.softmax
    ))
    
    model.summary()

    # ----- TRAIN ----- #

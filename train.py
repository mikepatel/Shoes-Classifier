"""
Michael Patel
June 2020

Project description:
    Binary shoe classifier between Nike and Adidas

File description:
    For preprocessing and model training
"""
################################################################################
# Imports
import os
import numpy as np
import shutil
import cv2

import tensorflow as tf


################################################################################
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_CHANNELS = 3

NUM_EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 1e-4

TRAIN_DIR = os.path.join(os.getcwd(), "data\\train")
TEST_DIR = os.path.join(os.getcwd(), "data\\test")
TEMP_DIR = os.path.join(os.getcwd(), "data\\temp")
SAVE_DIR = os.path.join(os.getcwd(), "saved_model")


################################################################################
# Main
if __name__ == "__main__":
    # remove 'saved_model' directory
    if os.path.exists(SAVE_DIR):
        shutil.rmtree(SAVE_DIR)

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
        rotation_range=20,
        width_shift_range=0.3,
        height_shift_range=0.3,
        brightness_range=[0.3, 1.0],
        shear_range=20,
        zoom_range=[0.7, 1.3],
        channel_shift_range=100.0,
        horizontal_flip=True,
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
    #quit()

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

    model.compile(
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        metrics=["accuracy"]
    )

    # ----- TRAIN ----- #
    history = model.fit(
        x=train_data_gen,
        steps_per_epoch=train_data_gen.samples / train_data_gen.batch_size,
        epochs=NUM_EPOCHS
    )

    # save model
    model.save(SAVE_DIR)

    # ----- TEST ----- #
    # load model
    model = tf.keras.models.load_model(SAVE_DIR)

    test_images = []
    for directory in os.listdir(TEST_DIR):
        for image_file in os.listdir(os.path.join(TEST_DIR, directory)):
            image_file_path = os.path.join(TEST_DIR, directory+"\\"+image_file)

            image = cv2.imread(image_file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
            image = np.array(image).astype(np.float32) / 255.0
            image = np.expand_dims(image, 0)  # shape: (1, WIDTH, HEIGHT, CHANNELS)

            prediction = model.predict(image)
            predict_name = int2class[int(np.argmax(prediction))]
            print(f'{image_file}: {predict_name}')

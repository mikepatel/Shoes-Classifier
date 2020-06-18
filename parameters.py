"""
Michael Patel
June 2020

Project description:
    Binary shoe classifier between Nike and Adidas

File description:
    For parameters
"""
################################################################################
# Imports
import os


################################################################################
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_CHANNELS = 3

NUM_EPOCHS = 200
BATCH_SIZE = 16
LEARNING_RATE = 1e-4

TRAIN_DIR = os.path.join(os.getcwd(), "data\\train")
TEST_DIR = os.path.join(os.getcwd(), "data\\test")
TEMP_DIR = os.path.join(os.getcwd(), "data\\temp")
SAVE_DIR = os.path.join(os.getcwd(), "saved_model")

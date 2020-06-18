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
SAVE_DIR = os.path.join(os.getcwd(), "saved_model")


################################################################################
# Main
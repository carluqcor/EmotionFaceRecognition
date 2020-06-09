import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import glob
import tensorflow as tf
import shutil

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, Input, GlobalMaxPooling2D, GlobalAveragePooling2D, Dropout, BatchNormalization, Flatten, Dense, GlobalMaxPooling3D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import layers, regularizers

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

import cv2
from tensorflow.keras.applications import vgg19

import os.path

import io

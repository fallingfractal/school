import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils
from keras.layers import Dropout, Flatten, Conv2D, MaxPooling2D


(X_train, y_train), (X_test, y_test) = mnist.load_data()

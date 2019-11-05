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
from PIL import Image
import requests
from io import BytesIO

imgWidth = 28
imgHeight = 28
(X_train, y_train), (X_test, y_test) = mnist.load_data()

num_pixels = imgWidth * imgHeight
X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')

#normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


#build the model
def baseline_model():
	# Create model
	model = Sequential()
	# hidden layer 784 input nodes
	model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
	# output layer , it has 10 outputs/nodes
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

model = baseline_model()

# Fit the model the actual training occurs here
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)



#save model for later predictions
model.save("model.h5")

#load model for testing
model = load_model("model.h5")















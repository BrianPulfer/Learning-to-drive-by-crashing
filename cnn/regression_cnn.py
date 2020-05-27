#!/home/usi/Desktop/cnn/venv/bin/python

# Standard library imports
import os

# Tensorflow imports
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping

# Numpy import
import numpy as np

# MatPlotLib import
import matplotlib.pyplot as plt

# Sci-Kit Learn import
from sklearn.utils import shuffle

# CV2
import cv2


START_IMAGE_NUMBER = 3
DATASET_SIZE = 10448 - 2448 #839 #1663 #1388 #783
#NUMBER_CLASSES = 5 #3
PATIENCE = 30

# Hyperparameters
IMAGES_TARGET_SIZE = (640/4, 400/4) #(160 ,100)
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NR_EPOCHS = 300

def get_data(percentage_train = 0.7):
	# Collecting the images
	images = []
	for i in range(START_IMAGE_NUMBER, START_IMAGE_NUMBER + DATASET_SIZE):
		cv2_img = cv2.imread(os.getcwd()+'/dataset/'+str(i)+".jpeg")
		cv2_img = cv2.resize(cv2_img, IMAGES_TARGET_SIZE)
		img = cv2_img[...,::-1].astype(np.float32) # Converts from GBR to RGB

		images.append(img_to_array(img))

	# Collecting the labels
	labels_file = open(os.getcwd()+'/dataset/labels.csv', 'r')
	labels = []
	for line in labels_file:
		label = line.split(',')[1]

		if label[0] == ' ':
			label.strip()

		labels.append(float(label))

	# Cutting the labels to the used ones
	labels = labels[:DATASET_SIZE]

	# Normalizing the labels in the range [-1, 1]
	max_label = max(max(labels), abs(min(labels)))
	labels = np.array(labels) / max_label

	

	images, labels = shuffle(images, labels)

	test_idx_start = int(len(labels) * percentage_train)

	x_train, y_train = tf.convert_to_tensor(images[:test_idx_start]), tf.convert_to_tensor(labels[:test_idx_start])
	x_test, y_test = tf.convert_to_tensor(images[test_idx_start:]), tf.convert_to_tensor(labels[test_idx_start:])

	# Normalizing pixels in range [0, 1]
	x_train, x_test = x_train / 255, x_test / 255

	# One-hot encoding labels (just in classification)
	# y_train, y_test = tf.one_hot(y_train, NUMBER_CLASSES), tf.one_hot(y_test, NUMBER_CLASSES)

	return (x_train, y_train), (x_test, y_test)
	

def create_model():
	model = Sequential()

	model.add(Conv2D(filters=8, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Conv2D(filters=16, kernel_size=(4, 4), strides=(2, 2), activation='relu')) # Added
	model.add(MaxPool2D(pool_size=(2, 2))) # Added
	model.add(Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), activation='relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dropout(rate=0.3))
	model.add(Dense(16, activation='tanh'))
	model.add(Dropout(rate=0.3))
	model.add(Dense(1, activation='tanh'))
	return model

def store(model):
	model.save('model.h5')

def plot_train_history(history):
	plt.plot(history['loss'])
	plt.plot(history['val_loss'])
	plt.title("Training History")
	plt.xlabel("Epochs")
	plt.ylabel("Loss")
	plt.legend(['Train', 'Validation'], loc='upper right')
	plt.show()

def main():
	# Collecting data set
	(x_train, y_train), (x_test, y_test) = get_data()
	
	# Creating model
	tf.random.set_seed(7)
	model = create_model()

	# Training and testing model
	model.compile(optimizer=Adam(learning_rate = LEARNING_RATE), loss='mean_squared_error', metrics=['accuracy'])
	train_history = model.fit(
		x_train, y_train, batch_size=BATCH_SIZE, epochs=NR_EPOCHS,
		validation_data=(x_test, y_test), callbacks=[EarlyStopping(monitor='val_loss', patience=PATIENCE)])
	
	# Storing the model
	store(model)

	# Plotting the training history
	plot_train_history(train_history.history)


if __name__ == '__main__':
	main()
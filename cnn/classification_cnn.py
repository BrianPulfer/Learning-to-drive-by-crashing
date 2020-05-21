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

DATASET_SIZE = 1216

PATIENCE = 30
NUMBER_CLASSES = 5

# Hyperparameters
IMAGES_TARGET_SIZE = (150, 150)
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NR_EPOCHS = 60

def get_data(percentage_train = 0.7):
	# Collecting the images
	images = []
	for i in range(1, DATASET_SIZE+1):
		img = load_img(os.getcwd()+'/dataset/'+str(i)+".jpeg", target_size=IMAGES_TARGET_SIZE)
		images.append(img_to_array(img))

	# Collecting the labels
	labels_file = open(os.getcwd()+'/dataset/labels.csv', 'r')
	labels = []

	line_counter = 0
	for line in labels_file:
		line_counter += 1
		if not line_counter > DATASET_SIZE:
			label = line.split(',')[1]
			label.strip()
			labels.append(int(label))

	images, labels = shuffle(images, labels)

	test_idx_start = int(len(labels) * percentage_train)

	x_train, y_train = tf.convert_to_tensor(images[:test_idx_start]), tf.convert_to_tensor(labels[:test_idx_start])
	x_test, y_test = tf.convert_to_tensor(images[test_idx_start:]), tf.convert_to_tensor(labels[test_idx_start:])

	# Normalizing pixels in range [0, 1]
	x_train, x_test = x_train / 255, x_test / 255

	# One-hot encoding labels
	y_train, y_test = tf.one_hot(y_train-1, NUMBER_CLASSES), tf.one_hot(y_test-1, NUMBER_CLASSES)

	return (x_train, y_train), (x_test, y_test)
	

def create_model():
	model = Sequential()

	model.add(Conv2D(filters=8, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), activation='relu')) # Added
	model.add(MaxPool2D(pool_size=(2, 2))) # Added
	model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation='relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dropout(rate=0.3))
	model.add(Dense(16, activation='tanh'))
	model.add(Dropout(rate=0.3))
	model.add(Dense(NUMBER_CLASSES, activation='softmax'))
	return model

def get_accuracy(targets, predictions):
	correct = 0

	for i in range(len(targets)):
		target, prediction = list(targets[i]), list(predictions[i])

		target_class = target.index(max(target))-1
		predicted_class = prediction.index(max(prediction))-1

		if target_class == predicted_class:
			correct = correct + 1

	return float(correct) / float(len(targets))

def store(model):
	model.save('model.h5')

def plot_train_history(history):
	plt.plot(history['loss'])
	plt.plot(history['val_loss'])
	plt.title('Training History')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend(['Train', 'Validation'], loc='upper right')
	plt.show()

def main():
	# Collecting data set
	(x_train, y_train), (x_test, y_test) = get_data()
	
	# Creating model
	tf.random.set_seed(7)
	model = create_model()

	# Training and testing model
	model.compile(optimizer=Adam(learning_rate = LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
	train_history = model.fit(
		x_train, y_train, batch_size=BATCH_SIZE, epochs=NR_EPOCHS,
		validation_data=(x_test, y_test), callbacks=[EarlyStopping(monitor='val_loss', patience=PATIENCE)])

	
	# Testing on the test data
	predictions = model.predict(x_test)

	# Printing the accuracy
	print("Accuracy: ", get_accuracy(y_test, predictions))
	
	# Storing the model
	store(model)

	# Plotting the training history
	plot_train_history(train_history.history)


if __name__ == '__main__':
	main()
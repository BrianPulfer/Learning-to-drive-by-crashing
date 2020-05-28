#!/home/usi/catkin_ws/src/driving_controller/src/venv/bin/python

# Importing random number generator
from random import random

# Importing rospy
import rospy

# Importing numpy
import numpy as np

# Importing tensorflow
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

# Importing the thymio controller
from controller import MightyThymioController

# Importing bridge to convert images from (640, 400) to any shape (e.g. (150, 150))
from cv_bridge import CvBridge, CvBridgeError
import cv2

# Maximum steer angle of thymio
MAX_STEER_ANGLE_SPEED = 2
ROTATE_SPEED = 0.05
ROTATE_CYCLES = 10
BIAS_CORRECTION = 0 #-0.24628784 (small) #-0.00720734 (regression 2) #-0.25224984 (regression 1)

# Speed of 0.1 and 80 Cycles make the thymio move of a 1 meter
FORWARD_SPEED = 0.1
MOVE_CYCLES = 10

PROCESSED_IMG_SIZE = (640/4, 400/4)

IMAGES_BRIDGE = CvBridge()

WALL_IN_FRONT_THRESHOLD = 0.8


def move_randomly(thymio):
	import random
	angle = 0.5-random.random()

	for i in range(MOVE_CYCLES):
		thymio.publish_speed(x=FORWARD_SPEED, y=0, z=angle)
	thymio.stop()

def to_input_image(camera_frame):
	"""
	cv2_img = IMAGES_BRIDGE.imgmsg_to_cv2(camera_frame, "rgb8")
	processed_cv2_img = cv2.resize(cv2_img, PROCESSED_IMG_SIZE) / 255
	input_image = tf.convert_to_tensor(img_to_array(processed_cv2_img))

	return input_image
	"""
	cv2_img = IMAGES_BRIDGE.imgmsg_to_cv2(camera_frame, "bgr8")
	cv2_img = cv2.resize(cv2_img, PROCESSED_IMG_SIZE)
	img = cv2_img[...,::-1].astype(np.float32) # Converts from GBR to RGB

	tensor = tf.convert_to_tensor(img_to_array(img))
	tensor = tensor / 255

	return tensor



def main():
	# Initializing Thymio
	thymio_name = rospy.get_param('~name')

	if thymio_name[0] == '/':
		thymio_name = thymio_name[1:]

	thymio = MightyThymioController(thymio_name, initialize_node=False)

	move_randomly(thymio)

	# Getting the model
	model = tf.keras.models.load_model('/home/usi/catkin_ws/src/driving_controller/src/model.h5')

	# Driving around
	while True:
		thymio_camera_frame = thymio.get_camera_frame()

		if thymio_camera_frame is not None:
			input_image = to_input_image(thymio_camera_frame)

			prediction = model.predict(np.array([input_image]))[0]
			print("Model Prediction: ", prediction)

			#random_deviation = (0.5 - random()) / 8 # [-0.0625, 0.0625]
			random_deviation = 0

			steer_angle, wall_is_in_front = prediction[0], prediction[1]

			if wall_is_in_front < WALL_IN_FRONT_THRESHOLD:
				for i in range(MOVE_CYCLES):
					thymio.publish_speed(x=FORWARD_SPEED, y=0, z=MAX_STEER_ANGLE_SPEED)
			else:
				for i in range(MOVE_CYCLES):
					thymio.publish_speed(x=FORWARD_SPEED, y=0, z=steer_angle)
			thymio.stop()


if __name__ == '__main__':
	rospy.init_node('driving_controller_node', anonymous=True)

	if rospy.has_param("~name"):
		main()
	else:
		print("Parameter name not inserted to run driving_controller")
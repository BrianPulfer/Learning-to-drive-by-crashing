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
MAX_STEER_ANGLE_SPEED = 0.25
ROTATE_SPEED = 0.05
ROTATE_CYCLES = 20
BIAS_CORRECTION = 0 #-0.00720734 #-0.25224984

# Speed of 0.1 and 20 Cycles make the thymio move of a 1/4 meter
FORWARD_SPEED = 0.1
MOVE_CYCLES = 20 * 2


def move_randomly(thymio):
	import random
	angle = 0.5-random.random()

	for i in range(MOVE_CYCLES):
		thymio.publish_speed(x=FORWARD_SPEED, y=0, z=angle)
	thymio.stop()


def main():
	# Initializing Thymio
	thymio_name = rospy.get_param('~name')

	if thymio_name[0] == '/':
		thymio_name = thymio_name[1:]

	thymio = MightyThymioController(thymio_name, initialize_node=False)

	move_randomly(thymio)

	# Getting the model
	model = tf.keras.models.load_model('/home/usi/catkin_ws/src/driving_controller/src/model.h5')

	# Creating a bridge for images
	images_bridge = CvBridge()

	# Driving around
	while True:
		thymio_camera_frame = thymio.get_camera_frame()

		if thymio_camera_frame is not None:
			cv2_img = images_bridge.imgmsg_to_cv2(thymio_camera_frame, "bgr8")
			processed_cv2_img = cv2.resize(cv2_img, (150, 150)) / 255

			input_image = tf.convert_to_tensor(img_to_array(processed_cv2_img))

			prediction = model.predict(np.array([input_image]))[0]
			print("Model Prediction: ", prediction)

			for i in range(MOVE_CYCLES):
				thymio.publish_speed(x=FORWARD_SPEED, y=0, z= -(prediction + BIAS_CORRECTION) * MAX_STEER_ANGLE_SPEED)
			thymio.stop()


if __name__ == '__main__':
	rospy.init_node('driving_controller_node', anonymous=True)

	if rospy.has_param("~name"):
		main()
	else:
		print("Parameter name not inserted to run driving_controller")
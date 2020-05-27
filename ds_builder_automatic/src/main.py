#!/usr/bin/env python

### IMPORTS

# Thymio controller (Moves thymio)
import controller

# Numpy (Computes dot products)
import numpy as np

# Rospy (Checks the parameter 'name' was passed)
import rospy

# CVBridge (Converts camera frames to jpeg images)
from cv_bridge import CvBridge, CvBridgeError
import cv2


def initialize_thymio():
	"""Given the argument 'name', creates and returns the thymio controller"""
	thymio_name = rospy.get_param('~name')

	if thymio_name[0] == '/':
		thymio_name = thymio_name[1:]

	thymio = controller.MightyThymioController(thymio_name, initialize_node=False)
	return thymio


def forward_until_approach_close(thymio, speed, sample_image_rate = 0.5):
	approaching = False

	while not rospy.is_shutdown() and not approaching:
		thymio.move_straight(speed=speed)

		if np.random.random() < sample_image_rate:
			store_one_more_image()

		for sensor in range(len(thymio.sensors)):
			if thymio.sensors[sensor][1] <= (thymio.sensors[sensor][2] + thymio.sensors[sensor][0])/2:
				approaching = True

	thymio.stop()


def store_image(thymio, image_path, label_path, image_count, bridge):
	"""Given a thymio, stores it's current camera frame with an appropriate label appending it to the specified file"""
	
	# Collecting the normalized sensors readings
	S = thymio.sensors[:5][:,1] / thymio.sensors[:5][:,2]

	# Deriving label and retrieving image
	label = np.dot([-2, -1, 0, 1, 2], S)
	image = thymio.get_camera_frame()

	if not image:
		return None

	# Storing the image
	cv2_img = bridge.imgmsg_to_cv2(image, "bgr8")
	cv2.imwrite(image_path, cv2_img)

	# Storing the label
	labels_file = open(label_path, 'a')
	labels_file.write(str(image_count)+', '+str(label)+'\n')
	labels_file.close()

	return label

def store_one_more_image():
	global THYMIO
	global IMAGE_COUNTER

	IMAGE_COUNTER += 1
	label = store_image(THYMIO, image_path=DATASET_PATH+str(IMAGE_COUNTER)+'.jpeg', label_path=LABELS_FILE_PATH, image_count=IMAGE_COUNTER, bridge=BRIDGE)

	return label

def rotate_by_pi(myt):
	# Sensors
	LEFT = 0
	CENTER_LEFT = 1
	CENTER = 2
	CENTER_RIGHT = 3
	RIGHT = 4
	REAR_LEFT = 5
	REAR_RIGHT = 6

	# Sensor's ranges
	MIN = 0
	CURRENT = 1
	MAX = 2

	# Rotating the Myt until it's givin the back to the approached object
	delta_angle = 0.1     # Radiants
	rot_speed = 0.1      # Rotational speed of each rotation message
	rot_threshold = 0.001 # Rotational threshold which determines rotation precision

	approach_sensor, approach_distance = None, 1000

	for sensor in range(5):
		if myt.sensors[sensor][CURRENT] < approach_distance:
			approach_sensor = sensor
			approach_distance = myt.sensors[sensor][CURRENT]

	rotating_left = not (approach_sensor == LEFT or approach_sensor == CENTER_LEFT)
	while not myt.is_approaching_something(front=False):
		angle = None

		if rotating_left:
			angle = myt.orientation.z + delta_angle
		else:
			angle = myt.orientation.z - delta_angle

		myt.rotate_in_place(angle, speed=rot_speed, threshold=rot_threshold, rotate_left=rotating_left)
		store_one_more_image()

	if myt.sensors[REAR_LEFT][CURRENT] <= myt.sensors[REAR_RIGHT][CURRENT]:
		while myt.sensors[REAR_LEFT][CURRENT] < myt.sensors[REAR_RIGHT][CURRENT]:
			myt.rotate_in_place(myt.orientation.z - delta_angle, speed=rot_speed, threshold=rot_threshold, rotate_left=rotating_left)
			store_one_more_image()
	else:
		while myt.sensors[REAR_LEFT][CURRENT] > myt.sensors[REAR_RIGHT][CURRENT]:
			myt.rotate_in_place(myt.orientation.z + delta_angle, speed=rot_speed, threshold=rot_threshold, rotate_left=rotating_left)
			store_one_more_image()

	myt.stop()



# Definitions
IMAGE_COUNTER = 0
DATASET_PATH = '/home/usi/catkin_ws/src/ds_builder_automatic/src/dataset/'
LABELS_FILE_PATH = DATASET_PATH + 'labels.csv'

THYMIO_SPEED = 0.3
BRIDGE = CvBridge()

THYMIO = None

def main():
	# Initializing Thymio
	global THYMIO
	THYMIO = initialize_thymio()

	while not rospy.is_shutdown():
		# Going forward until something is approached
		forward_until_approach_close(THYMIO, THYMIO_SPEED)

		# Storing the image with the label
		label = store_one_more_image()

		# If the image was stored successfully
		if label: 
			# Steer the thymio
			rotate_by_pi(THYMIO)
			"""
			angle = np.arcsin(2*thymio.orientation.z) + label * np.pi / 12
			print('Label is: ', label)
			thymio.teleport(thymio.point.x, thymio.point.y, np.sin(angle/2), np.cos(angle/2))
			"""


if __name__ == '__main__':
	rospy.init_node('ds_builder_automatic_node', anonymous=True)

	if not rospy.has_param('~name'):
		print('Script need thymio name to run properly')
	else:
		main()

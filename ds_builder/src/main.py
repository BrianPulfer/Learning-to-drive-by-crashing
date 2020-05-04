#!/usr/bin/env python

import rospy
import controller

from cv_bridge import CvBridge, CvBridgeError
import cv2

import random

def ask_user_to_move_thymio(thymio):
	print("Pick MightyThymio x position (between -5 and 5):")
	x = input()

	print("Pick MightyThymio y position (between -5 and 5):")
	y = input()

	print("Pick MightyThymio theta angle (between -1 and 1):")
	theta = input()
	thymio.teleport(x, y, theta)


def ask_user_to_label_thymio_view(thymio):
	print("Please insert label for this picture")
	label = input()
	return label


def move_thymio_randomly(thymio, min_x, max_x, min_y, max_y, theta = None):
	x, y = random.uniform(min_x, max_x), random.uniform(min_y, max_y)

	if theta is None:
		theta = random.uniform(-1, 1)

	thymio.teleport(x, y, theta)


def store_in_dataset(image, label, path, filename, bridge):
	cv2_img = bridge.imgmsg_to_cv2(image, "bgr8")
	cv2.imwrite(path + filename + '.jpeg', cv2_img)

	labels_file = open(path+'labels.csv', 'a')
	labels_file.write(filename+', '+str(label)+'\n')
	labels_file.close()


def main():
	# Initializing Thymio
	thymio_name = rospy.get_param('~name')

	if thymio_name[0] == '/':
		thymio_name = thymio_name[1:]

	thymio = controller.MightyThymioController(thymio_name, initialize_node=False)

	# Picking a directory to store the dataset
	dataset_path = '/home/usi/catkin_ws/src/ds_builder/src/dataset/'

	# Creating the bridge to convert raw images into jpg
	bridge = CvBridge()

	# Creating a counter to give different file names
	counter = 1

	# Places to avoid while labeling data (because obstacles are there)
	places_to_avoid = [
	[-3, -3],
	[-3, 2],
	[-3, 3],
	[-2, -2],
	[-2, 2],
	[-2, 3],
	[2, -1],
	[2, 1],
	[2, 2],
	[3, 1],
	[3, 2]
	]

	# Gathering the data
	while not rospy.is_shutdown():
		# Put thymio in 9*9 positions each with 10 different angles (collect 9*9*10= 810 images)
		for x in [-4, -3, -2, -1, 0, 1, 2, 3, 4]:
			for y in [-4, -3, -2, -1, 0, 1, 2, 3, 4]:
				for theta in [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]:
					if [x, y] not in places_to_avoid:
						print("Teleporting the Thymio to: ", x, y, theta)

						thymio.teleport(x, y, theta, w=0.5)
						label = ask_user_to_label_thymio_view(thymio)
						store_in_dataset(thymio.get_camera_frame(), label, dataset_path, str(counter), bridge)

						counter += 1


if __name__ == '__main__':
	rospy.init_node('ds_builder_node', anonymous=True)

	if not rospy.has_param('~name'):
		print('Script need thymio name to run properly')
	else:
		main()
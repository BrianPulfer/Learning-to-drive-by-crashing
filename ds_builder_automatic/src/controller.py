#!/usr/bin/env python
import sys

import rospy
from geometry_msgs.msg import Point, Twist, PoseWithCovariance, TwistWithCovariance, Quaternion
from sensor_msgs.msg import Range
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry

import numpy as np
import math

from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState

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

class MightyThymioController:

	def __init__(self, thymio_name, initialize_node=True, initialize_name = ''):

		# Initializing node
		if initialize_node:
			rospy.init_node(initialize_name, anonymous = True)

		# Initializing name
		self.name = thymio_name
		self.rate = rospy.Rate(10)

		# Initializing position and angle
		self.point = Point()
		self.orientation = Quaternion()

		# Initializing initial ranges (sensors)
		self.sensors = np.zeros((7, 3))

		# Initializing the velocity publisher
		self.velocity_publisher = rospy.Publisher('/'+self.name+'/cmd_vel', Twist, queue_size=10)

		# Initializing odometry subscriber
		self.odometry_subscriber = rospy.Subscriber('/'+self.name+'/odom', Odometry, self.odometry_callback)

		# Initializing current image
		self.camera_frame = None

		# Initializing the sensors (front) subscribers
		rospy.Subscriber('/'+self.name+'/proximity/left', Range, self.sensors_callback)
		rospy.Subscriber('/'+self.name+'/proximity/center_left', Range, self.sensors_callback)
		rospy.Subscriber('/'+self.name+'/proximity/center', Range, self.sensors_callback)
		rospy.Subscriber('/'+self.name+'/proximity/center_right', Range, self.sensors_callback)
		rospy.Subscriber('/'+self.name+'/proximity/right', Range, self.sensors_callback)

		# Initializing the sensors (front) subscribers
		rospy.Subscriber('/'+self.name+'/proximity/rear_left', Range, self.sensors_callback)
		rospy.Subscriber('/'+self.name+'/proximity/rear_right', Range, self.sensors_callback)

		# Initializing camera subscriber
		rospy.Subscriber('/'+self.name+'/camera/image_raw', Image, self.camera_callback)

		# When rospy shutdowns, stop the thymio
		rospy.on_shutdown(self.stop)

	def euclidean_distance(self, goal_position):
		return np.sqrt((self.point.x - goal_position.x)**2 + (self.point.y - goal_position.y)**2)

	def teleport(self, x, y, theta, w=1):
		"""Teleports the Robot at position (x, y, 0) with orientation (0, 0, theta) """
		state_msg = ModelState()

		state_msg.model_name = '/'+self.name
		
		state_msg.pose.position.x = float(x)
		state_msg.pose.position.y = float(y)
		state_msg.pose.position.z = 0

		state_msg.pose.orientation.x = 0
		state_msg.pose.orientation.y = 0
		state_msg.pose.orientation.z = float(theta)
		state_msg.pose.orientation.w = w

		rospy.wait_for_service('/gazebo/set_model_state')

		try:
			set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
			return set_state(state_msg)
		except rospy.ServiceException, e:
			print(e)
		return None

	def set_camera_angle(self, angle):
		pass

	def get_camera_frame(self):
		return self.camera_frame

	def camera_callback(self, data):
		self.camera_frame = data

	def sensors_callback(self, data):
		#rospy.loginfo('Received update from sensors')
		"""
		DATA IN A 'RANGE' OBJECT:

            uint8 ULTRASOUND=0
            uint8 INFRARED=1
            std_msgs/Header header
            uint8 radiation_type
            float32 field_of_view
            float32 min_range
            float32 max_range
            float32 range
		"""
		rng, max_rng, min_rng = data.range, data.max_range, data.min_range
		header_str = data.header.frame_id

		if 'center' in header_str:
			if 'left' in header_str:
				self.sensors[CENTER_LEFT][CURRENT], self.sensors[CENTER_LEFT][MAX], self.sensors[CENTER_LEFT][MIN] = rng, max_rng, min_rng
			elif 'right' in header_str:
				self.sensors[CENTER_RIGHT][CURRENT], self.sensors[CENTER_RIGHT][MAX], self.sensors[CENTER_RIGHT][MIN] = rng, max_rng, min_rng
			else:
				self.sensors[CENTER][CURRENT], self.sensors[CENTER][MAX], self.sensors[CENTER][MIN] = rng, max_rng, min_rng
		elif 'rear' in header_str:
			if 'left' in header_str:
				self.sensors[REAR_LEFT][CURRENT], self.sensors[REAR_LEFT][MAX], self.sensors[REAR_LEFT][MIN] = rng, max_rng, min_rng
			else:
				self.sensors[REAR_RIGHT][CURRENT], self.sensors[REAR_RIGHT][MAX], self.sensors[REAR_RIGHT][MIN] = rng, max_rng, min_rng
		elif 'left' in header_str:
			self.sensors[LEFT][CURRENT], self.sensors[LEFT][MAX], self.sensors[LEFT][MIN] = rng, max_rng, min_rng
		else:
			self.sensors[RIGHT][CURRENT], self.sensors[RIGHT][MAX], self.sensors[RIGHT][MIN] = rng, max_rng, min_rng


	def odometry_callback(self, data):
		self.point = data.pose.pose.position
		self.orientation = data.pose.pose.orientation

	def move_straight(self, speed = 0.1):
		msg = Twist()
		msg.linear.x = speed
		self.velocity_publisher.publish(msg)
		self.rate.sleep()

	def is_approaching_something(self, front=True):
		start_sensor, end_sensor = None, None
		if front:
			start_sensor, end_sensor = 0, 5
		else:
			start_sensor, end_sensor = 5, 7

		for i in range(start_sensor, end_sensor):
			if self.sensors[i][CURRENT] < self.sensors[i][MAX]:
				return True
		return False

	def rotate_in_place(self, final_angle, rotate_left=True, speed=0.5, threshold=0.0001, break_on_exceed=True, radiants=True):
		if not rotate_left:
			speed *= -1

		msg = Twist()
		msg.linear.x, msg.angular.z = 0, speed

		# Z=0 -> 0 deg, Z=0.5 = 90 deg, Z = 1 -> 180 deg, Z = -0.5 -> -90 deg (every spin on itself, sign is inverted)
		if radiants:
			final_angle = final_angle / (np.pi)

		last_distance = abs(self.orientation.z - final_angle)
		new_distance = abs(self.orientation.z - final_angle)

		while not rospy.is_shutdown():
			self.velocity_publisher.publish(msg)
			self.rate.sleep()

			last_distance = new_distance
			new_distance = abs(self.orientation.z - final_angle)

			if new_distance > last_distance and break_on_exceed:
				break

			if last_distance < threshold:
				break

	def stop(self):
		msg = Twist()
		msg.linear.x = 0
		msg.linear.y = 0
		msg.angular.z = 0
		self.velocity_publisher.publish(msg)
		self.rate.sleep()

	def publish_speed(self, x, y, z):
		msg = Twist()
		msg.linear.x = x
		msg.linear.y = y
		msg.angular.z = z
		self.velocity_publisher.publish(msg)
		self.rate.sleep()

	def do_an_eight(self, speed = 0.1, angle=np.pi/8, threshold = 0.01):
		msg = Twist()
		msg.linear.x = speed
		msg.angular.z = angle

		for i in range(2):
			start_position = self.point

			# Starting the circle (getting away from threshold)
			while not rospy.is_shutdown() and self.euclidean_distance(start_position) < threshold:
				self.velocity_publisher.publish(msg)

			# Closing the circle (approaching the threshold)
			while not rospy.is_shutdown() and self.euclidean_distance(start_position) > threshold:
				self.velocity_publisher.publish(msg)

			angle = -2 * angle
			msg.angular.z = angle

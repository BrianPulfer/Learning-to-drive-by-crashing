# Project
This project represents the final project of the Robotics course held in the spring semester 2020 by the University of Southern Switzerland (USI). <br/><br/>

The goal of the project is to teach a robot (the Mighty Thymio, a personalized variant of the Thymio by IDSIA) not to crash against obstacles in an environment by using a convolutional neural network that given an image, predicts a steering angle for the robot that travels at constant speed.

## Content
### ds_builder/
This folder contains a program that allow to teleport the robot to any position (x, y) on the plane with any orientation (rotation on z-axis) and store and label what is beign currently seen by the robot (camera frame).

### worlds/
This folder contains the different environments the robot can be located in.

### cnn/
Contains the script and the dataset to create the Convolutional Neural Network model. The model is stored in an .h5 file as well.

### driving_controller/
Contains the script that uses the convolutional neural network to control the thymio. Also, contains the .launch file.

# Authors
Brian Pulfer, Rwiddhi Chakraborty, Shubhayu Das

#! /usr/bin/env morseexec

""" Basic MORSE simulation scene for <Rat_Experiment_1> environment

Feel free to edit this template as you like!
"""

from morse.builder import *

# Add the MORSE mascott, MORSY.
# Out-the-box available robots are listed here:
# http://www.openrobots.org/morse/doc/stable/components_library.html
#
# 'morse add robot <name> Rat_Experiment_1' can help you to build custom robots.
robot = Morsy()

# The list of the main methods to manipulate your components
# is here: http://www.openrobots.org/morse/doc/stable/user/builder_overview.html
robot.translate(1.0, 0.0, 0.0)

# Add a motion controller
# Check here the other available actuators:
# http://www.openrobots.org/morse/doc/stable/components_library.html#actuators
#
# 'morse add actuator <name> Rat_Experiment_1' can help you with the creation of a custom
# actuator.
motion = MotionVW()
robot.append(motion)



camera = VideoCamera()
camera.translate(x=0.2, z=1.6)
camera.properties(cam_focal=30.0)
camera.properties(cam_width = 256)
camera.properties(cam_height = 256)
robot.append(camera)


# Add a keyboard controller to move the robot with arrow keys.
keyboard = Keyboard()
robot.append(keyboard)
keyboard.properties(ControlType = 'Position')


laserscanner = Sick()
laserscanner.translate(x=0,z=0.1)
laserscanner.properties(scan_window=360.0, resolution= 30.0)
robot.append(laserscanner)

# Add a pose sensor that exports the current location and orientation
# of the robot in the world frame
# Check here the other available actuators:
# http://www.openrobots.org/morse/doc/stable/components_library.html#sensors
#
# 'morse add sensor <name> Rat_Experiment_1' can help you with the creation of a custom
# sensor.
pose = Pose()
robot.append(pose)

# To ease development and debugging, we add a socket interface to our robot.
#
# Check here: http://www.openrobots.org/morse/doc/stable/user/integration.html 
# the other available interfaces (like ROS, YARP...)
robot.add_default_interface('socket')


# set 'fastmode' to True to switch to wireframe mode
env = Environment('room4.blend', fastmode = False)
env.set_camera_location([10.0, -10.0, 10.0])
env.set_camera_rotation([1.05, 0, 0.78])


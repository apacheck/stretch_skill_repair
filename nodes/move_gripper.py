#!/usr/bin/env python

from __future__ import print_function

from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

import rospy
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction
from control_msgs.msg import FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint

from visualization_msgs.msg import MarkerArray, Marker

from geometry_msgs.msg import PointStamped

from sensor_msgs.msg import PointCloud2

from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse

import math
import time
import threading
import sys

import tf2_ros
import argparse as ap
import numpy as np
import threading
import ros_numpy as rn

import hello_helpers.hello_misc as hm
# import stretch_funmap.navigate as nv


class MoveGripper(hm.HelloNode):
    def __init__(self):
        hm.HelloNode.__init__(self)
        self.rate = 10.0
        self.joint_states = None
        self.joint_states_lock = threading.Lock()

        marker = Marker()
        self.mouth_marker_type = marker.CUBE
        self.mouth_point = None

        num_pan_angles = 5

        # looking out along the arm
        middle_pan_angle = -math.pi/2.0

        look_around_range = math.pi/3.0
        min_pan_angle = middle_pan_angle - (look_around_range / 2.0)
        max_pan_angle = middle_pan_angle + (look_around_range / 2.0)
        pan_angle = min_pan_angle
        pan_increment = look_around_range / float(num_pan_angles - 1.0)
        self.pan_angles = [min_pan_angle + (i * pan_increment)
                           for i in range(num_pan_angles)]
        self.pan_angles = self.pan_angles + self.pan_angles[1:-1][::-1]
        self.prev_pan_index = 0

        self.move_lock = threading.Lock()

        with self.move_lock:
            self.handover_goal_ready = False

    def do_something(self):
        #  do something
        print("doing something")
        hm.HelloNode.main(self, 'move_gripper', 'move_gripper', wait_for_first_pointcloud=False)
        print("Doing something again")
        with self.move_lock:
            pan_index = np.random.randint(len(self.pan_angles))
            pan_angle = self.pan_angles[pan_index]
            pose = {'joint_head_pan': -1.57,
            'joint_head_tilt': -0.2,
            'wrist_extension': 0.2,
            'joint_lift': 0.7,
            'joint_wrist_yaw': 1.570796327}
            # 'gripper_aperture': 0.04}
            # pose={'joint_wrist_yaw': 0}
            # pose={'gripper_aperture': 0.05}
            # pose={'joint_head_pan': pan_angle}
            self.move_to_pose(pose)
            self.prev_pan_index = pan_index


if __name__ == '__main__':
    try:
        parser = ap.ArgumentParser(description='Handover an object.')
        args, unknown = parser.parse_known_args()
        node = MoveGripper()
        node.do_something()
    except KeyboardInterrupt:
        rospy.loginfo('interrupt received, so shutting down')

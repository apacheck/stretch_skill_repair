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

        self.move_lock = threading.Lock()

        with self.move_lock:
            self.handover_goal_ready = False

    def follow_trajectory(self, data):
        # Data should be a numpy array with x, y, theta, wrist_extension, z, wrist_theta
        rospy.loginfo("Starting follow_trajectory")
        hm.HelloNode.main(self, 'move_gripper', 'move_gripper', wait_for_first_pointcloud=False)
        with self.move_lock:
            for d in data:
                rospy.loginfo("Moving gripper and arm to: extension: {:.2f}, lift: {:.2f}, rotation: {:.2f}".format(d[3], d[4], d[5]))
                pose = {
                'wrist_extension': d[3],
                'joint_lift': d[4],
                'joint_wrist_yaw': d[5]
                # 'rotate_mobile_base': 0.1
                # 'translate_mobile_base': -0.2
                }
                # translate_mobile_base
                # rotate_mobile_base
                self.move_to_pose(pose)
                break
                # rospy.sleep(0.5)
        rospy.loginfo("Completed follow_trajectory")


if __name__ == '__main__':
    try:
        parser = ap.ArgumentParser(description='Handover an object.')
        args, unknown = parser.parse_known_args()
        node = MoveGripper()
        data = np.loadtxt('/home/adam/repos/synthesis_based_repair/data/stretch/trajectories/skillStretch2/train/rollout-0.txt')
        node.follow_trajectory(data)
    except KeyboardInterrupt:
        rospy.loginfo('interrupt received, so shutting down')

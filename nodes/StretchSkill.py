#!/usr/bin/env python

from __future__ import print_function

from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

import rospy
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction
from control_msgs.msg import FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory

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
import moveit_commander
import moveit_msgs.msg
from moveit_commander.conversions import pose_to_list
# import stretch_funmap.navigate as nv

IS_SIM = True
ORIGIN_FRAME = 'odom'
STRETCH_FRAME = 'base_link'
CMD_VEL_TOPIC = '/stretch_diff_drive_controller/cmd_vel'
DO_THETA_CORRECTION = True
DO_MOVE_ARM = True
DO_MOVE_GRIPPER = True

# IS_SIM = False
# ORIGIN_FRAME = 'origin'
# STRETCH_FRAME = 'stretch'
# CMD_VEL_TOPIC = '/stretch/cmd_vel'

class StretchSkill(hm.HelloNode):
    def __init__(self):
        if IS_SIM:
            moveit_commander.roscpp_initialize(sys.argv)
            rospy.init_node('controller', anonymous=True)
        else:
            hm.HelloNode.__init__(self)
            hm.HelloNode.main(self, 'move_gripper', 'move_gripper', wait_for_first_pointcloud=False)
        self.rate = rospy.Rate(20.0)
        self.joint_states = None

        # For use with mobile base control
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.vel_pub = rospy.Publisher(CMD_VEL_TOPIC, Twist, queue_size=10)

        if not IS_SIM:
            self.joint_states_lock = threading.Lock()

            self.move_lock = threading.Lock()

            with self.move_lock:
                self.handover_goal_ready = False


    def open_gripper(self):
        if DO_MOVE_GRIPPER:
            rospy.loginfo("Opening gripper")
            if not IS_SIM:
                with self.move_lock:
                    pose = {
                    'gripper_aperture': 0.05
                    }
                    self.move_to_pose(pose)
            else:
                move_group_hand = moveit_commander.MoveGroupCommander("stretch_gripper")
                joint_goal = move_group_hand.get_current_joint_values()
                print(joint_goal)
                joint_goal[0] = 0.2
                joint_goal[1] = 0.2
                move_group_hand.go(joint_goal, wait=True)
                joint_goal = move_group_hand.get_current_joint_values()
                print(joint_goal)
        else:
            rospy.loginfo("Global parameter set to not move gripper")

    def close_gripper(self):
        if DO_MOVE_GRIPPER:

            rospy.loginfo("Closing gripper")

            if not IS_SIM:
                with self.move_lock:
                    pose = {
                    'gripper_aperture': -0.03
                    }
                    self.move_to_pose(pose)
            else:
                move_group_hand = moveit_commander.MoveGroupCommander("stretch_gripper")
                joint_goal = move_group_hand.get_current_joint_values()
                print(joint_goal)
                joint_goal[0] = 0
                joint_goal[1] = 0
                move_group_hand.go(joint_goal, wait=True)
                joint_goal = move_group_hand.get_current_joint_values()
                print(joint_goal)
        else:
            rospy.loginfo("Global parameter set to not move gripper")

    def findPose(self, frame):
        found_transform = False
        while not found_transform:
            try:
                trans_stamped = self.tfBuffer.lookup_transform(ORIGIN_FRAME, frame, rospy.Time.now(), rospy.Duration(1.0))
                trans = trans_stamped.transform
                found_transform = True
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                self.rate.sleep()
                continue

        return trans


    def follow_trajectory(self, data):
        # Data should be a numpy array with x, y, theta, wrist_extension, z, wrist_theta
        rospy.loginfo("Starting follow_trajectory")
        if IS_SIM:
            robot = moveit_commander.RobotCommander()
            # rospy.sleep(10.0)
            move_group_arm = moveit_commander.MoveGroupCommander("stretch_arm")

        close_enough = 0.001
        epsilon = 0.1
        maxV = 2
        wheel2center = 0.1778

        for d in data:

            dl = d.tolist()
            if DO_MOVE_ARM:
                rospy.loginfo("Moving gripper and arm to: extension: {:.2f}, lift: {:.2f}, rotation: {:.2f}".format(d[3], d[4], d[5]))
                if not IS_SIM:
                    with self.move_lock:
                        pose = dict()
                        if d[3] != -10:
                            pose['wrist_extension'] = d[3]
                        if d[4] != -10:
                            pose['joint_lift'] = d[4]
                        if d[5] != -10:
                            pose['joint_wrist_yaw'] = d[5]
                        # 'rotate_mobile_base': 0.1
                        # 'translate_mobile_base': -0.2

                        # The base can also be controlled with odometery?
                        # translate_mobile_base
                        # rotate_mobile_base
                        self.move_to_pose(pose)
                else:
                    joint_goal = move_group_arm.get_current_joint_values()
                    if dl[4] != -10:
                        joint_goal[0] = dl[4]
                    if dl[3] != -10:
                        joint_goal[1] = dl[3]/4
                        joint_goal[2] = dl[3]/4
                        joint_goal[3] = dl[3]/4
                        joint_goal[4] = dl[3]/4
                    if dl[5] != -10:
                        joint_goal[5] = dl[5]

                    move_group_arm.go(joint_goal, wait=True)

            rospy.loginfo("Moving base to: x: {:.2f}, y: {:.2f}, theta: {:.2f}".format(d[0], d[1], d[2]))

            # Control the mobile base with feedback linearization and visiting waypoint
            # rospy.loginfo("Beginning feedback linearization for visit waypoint")
            at_waypoint = False
            if d[0] == -10:
                at_waypoint = True
            while not at_waypoint:
                trans_stretch = self.findPose(STRETCH_FRAME)

                dist_to_waypoint = np.sqrt([np.square(d[0] - trans_stretch.translation.x) +
                                            np.square(d[1] - trans_stretch.translation.y)])
                # rospy.loginfo("Distance to waypoint: {}".format(dist_to_waypoint))

                if dist_to_waypoint < close_enough:
                    at_waypoint = True

                if not at_waypoint:
                    cmd_vx, cmd_vy, theta = findDeltaPose(trans_stretch, d)
                    cmd_v, cmd_w = feedbackLin(cmd_vx, cmd_vy, theta, epsilon)
                    cmd_v, cmd_w = thresholdVel(cmd_v, cmd_w, maxV, wheel2center)
                    # _, cmd_v, cmd_w = calc_control_command(cmd_vx, cmd_vy, theta, epsilon)

                    # rospy.loginfo("Robot is at: x: {:.3f}, y: {:.3f}, z: {:.3f}, theta: {:.3f}".format(trans_stretch.translation.x, trans_stretch.translation.y, trans_stretch.translation.z, theta))
                    # rospy.loginfo("cmd_v: {} cmd_w: {}".format(cmd_v, cmd_w))

                    vel_msg = Twist()
                    vel_msg.linear.x = cmd_v
                    vel_msg.angular.z = cmd_w
                    self.vel_pub.publish(vel_msg)
                    self.rate.sleep()

            # Rotate if the current and desired theta do not match
            # rospy.loginfo("Beginning theta correction")
            # rospy.loginfo("Goal theta: {}".format(d[2]))
            make_theta_correction = True
            while make_theta_correction and DO_THETA_CORRECTION and d[2] != -10:
                trans_stretch = self.findPose(STRETCH_FRAME)

                _, _, theta = findDeltaPose(trans_stretch, d)
                # theta += np.pi
                # print("Real theta: {}".format(theta))
                if theta < np.pi:
                    theta += 2 * np.pi
                if theta > np.pi:
                    theta -= 2 * np.pi
                # rospy.loginfo("Robot is at: x: {:.3f}, y: {:.3f}, z: {:.3f}, theta: {:.3f}".format(trans_stretch.translation.x, trans_stretch.translation.y, trans_stretch.translation.z, theta))

                cmd_w = d[2] - theta
                if cmd_w > np.pi:
                    cmd_w -= 2 * np.pi
                if cmd_w < -np.pi:
                    cmd_w += 2 * np.pi
                if np.abs(cmd_w) < 0.0005:
                    make_theta_correction = False
                else:
                    # rospy.loginfo("cmd_v: {} cmd_w: {}".format(0, cmd_w))
                    vel_msg = Twist()
                    vel_msg.angular.z = cmd_w
                    self.vel_pub.publish(vel_msg)
                    self.rate.sleep()

            rospy.loginfo("Verifying new pose")
            found_transform = False
            trans_stretch = self.findPose(STRETCH_FRAME)

            rospy.loginfo("Robot is at: x: {:.3f}, y: {:.3f}, z: {:.3f}, theta: {:.3f}".format(trans_stretch.translation.x, trans_stretch.translation.y, trans_stretch.translation.z, theta))

            trans_ee = self.findPose('link_grasp_center')

            rospy.loginfo("Gripper is at: x: {:.3f}, y: {:.3f}, z: {:.3f}".format(trans_ee.translation.x, trans_ee.translation.y, trans_ee.translation.z))

        rospy.loginfo("Completed follow_trajectory")
        if IS_SIM and DO_MOVE_ARM:
            move_group_arm.stop()

def feedbackLin(arg_cmd_vx, arg_cmd_vy, arg_theta, arg_epsilon):
    cmd_vi = np.array([[arg_cmd_vx], [arg_cmd_vy]])
    R_b_i = np.array([[np.cos(arg_theta), np.sin(arg_theta)], [-np.sin(arg_theta), np.cos(arg_theta)]])
    cmd_vw = np.dot(np.dot(np.array([[1, 0], [0, 1/arg_epsilon]]), R_b_i), cmd_vi)
    cmd_v = cmd_vw[0]
    cmd_w = cmd_vw[1]

    return cmd_v, cmd_w

def thresholdVel(arg_cmd_v, arg_cmd_w, arg_maxV, arg_wheel2center):
    maxW = arg_maxV/arg_wheel2center
    ratioV = np.abs(arg_cmd_v/arg_maxV)
    ratioW = np.abs(arg_cmd_w/maxW)
    ratioTot = ratioV + ratioW
    if ratioTot > 1:
        arg_cmd_v = arg_cmd_v/ratioTot
        arg_cmd_w = arg_cmd_w/ratioTot

    return arg_cmd_v, arg_cmd_w


def findDeltaPose(arg_cur_pose, arg_desired_pose):
    cmd_vx = arg_desired_pose[0] - arg_cur_pose.translation.x
    cmd_vy = arg_desired_pose[1] - arg_cur_pose.translation.y
    q1 = arg_cur_pose.rotation.x
    q2 = arg_cur_pose.rotation.y
    q3 = arg_cur_pose.rotation.z
    q0 = arg_cur_pose.rotation.w
    theta = np.arctan2(2 * (q1 * q2 + q0 * q3), q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 **2)
    if not IS_SIM:
        theta -= np.pi
    return cmd_vx, cmd_vy, theta


if __name__ == '__main__':
    try:
        parser = ap.ArgumentParser(description='Handover an object.')
        args, unknown = parser.parse_known_args()
        node = StretchSkill()
        data4 = np.loadtxt('/home/adam/repos/synthesis_based_repair/data/stretch/trajectories/skillStretch4/train/rollout-0.txt')
        data5 = np.loadtxt('/home/adam/repos/synthesis_based_repair/data/stretch/trajectories/skillStretch5/train/rollout-0.txt')
        data6 = np.loadtxt('/home/adam/repos/synthesis_based_repair/data/stretch/trajectories/skillStretch6/train/rollout-0.txt')
        data7 = np.loadtxt('/home/adam/repos/synthesis_based_repair/data/stretch/trajectories/skillStretch7/train/rollout-0.txt')

        unit_box_pose = node.findPose('unit_box')
        stretch_box_pose = np.array([[unit_box_pose.translation.x, unit_box_pose.translation.y - 0.75, np.pi, -10, -10, -10]])
        node.open_gripper()
        rospy.loginfo("Moving to box")
        # node.follow_trajectory(stretch_box_pose)
        rospy.loginfo("Picking up box")
        node.follow_trajectory(data7)
        # node.follow_trajectory(data5)
        # node.follow_trajectory(data6)
        node.close_gripper()
    except KeyboardInterrupt:
        rospy.loginfo('interrupt received, so shutting down')

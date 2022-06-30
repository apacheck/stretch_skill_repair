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
from synthesis_based_repair.tools import write_spec, clear_file, dict_to_formula, json_load_wrapper

from dl2_lfd.nns.dmp_nn import DMPNN
from dl2_lfd.dmps.dmp import load_dmp_demos, DMP
import torch
from dl2_lfd.helper_funcs.conversions import np_to_pgpu
from gazebo_ros_link_attacher.srv import Attach, AttachRequest, AttachResponse

DEVICE="cpu"

# import stretch_funmap.navigate as nv

IS_SIM = True
if IS_SIM:
    ORIGIN_FRAME = 'odom'
    STRETCH_FRAME = 'robot::base_link'
    CMD_VEL_TOPIC = '/stretch_diff_drive_controller/cmd_vel'
    DO_THETA_CORRECTION = False
    DO_MOVE_GRIPPER = True
    OBJ_NAME = 'box_1::link'
else:
    IS_SIM = False
    ORIGIN_FRAME = 'origin'
    STRETCH_FRAME = 'stretch'
    CMD_VEL_TOPIC = '/stretch/cmd_vel'
    DO_THETA_CORRECTION = False
    DO_MOVE_GRIPPER = True
    OBJ_NAME = 'duck'

class StretchSkill(hm.HelloNode):
    def __init__(self):
        rospy.loginfo("Creating stretch skill")
        if IS_SIM:
            moveit_commander.roscpp_initialize(sys.argv)
            rospy.init_node('controller', anonymous=True)
            robot = moveit_commander.RobotCommander()
            # rospy.sleep(10.0)
            self.move_group_arm = moveit_commander.MoveGroupCommander("stretch_arm")
            self.attach_srv = rospy.ServiceProxy('/link_attacher_node/attach',
                                 Attach)
            self.attach_srv.wait_for_service()
            self.detach_srv = rospy.ServiceProxy('/link_attacher_node/detach',
                                 Attach)
            self.detach_srv.wait_for_service()
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

    def openGripper(self):
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

    def closeGripper(self):
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

    def attachObject(self, obj_name):
        rospy.loginfo("Attaching gripper and {}".format(obj_name))
        req = AttachRequest()
        req.model_name_1 = "robot"
        req.link_name_1 = "link_gripper_finger_left"
        req.model_name_2 = obj_name
        req.link_name_2 = "link"

        self.attach_srv.call(req)

    def detachObject(self, obj_name):
        rospy.loginfo("Detaching gripper and {}".format(obj_name))
        req = AttachRequest()
        req.model_name_1 = "robot"
        req.link_name_1 = "link_gripper_finger_left"
        req.model_name_2 = obj_name
        req.link_name_2 = "link"

        self.detach_srv.call(req)

    def findPose(self, frame):
        #TODO Change this to not be a class function, but pass in the transform or something?
        found_transform = False
        cnt = 1
        while not found_transform:
            try:
                trans_stamped = self.tfBuffer.lookup_transform(ORIGIN_FRAME, frame, rospy.Time.now(), rospy.Duration(1.0))
                trans = trans_stamped.transform
                found_transform = True
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                self.rate.sleep()
                cnt += 1
                if cnt % 2 == 0:
                    rospy.loginfo("Can't find transform")
                continue

        return trans

    def getJointValues(self):
        # Returns extension, lift, wrist yaw
        if IS_SIM:
            move_group_arm = moveit_commander.MoveGroupCommander("stretch_arm")
            joint_values = move_group_arm.get_current_joint_values()
        else:
            rospy.loginfo("Not implemented")
        return np.array([np.sum(joint_values[1:5]), joint_values[0], joint_values[5]])

    def followTrajectory(self, data):
        # Data should be a numpy array with x, y, theta, wrist_extension, z, wrist_theta
        rospy.loginfo("Starting followTrajectory")

        for d in data:

            if DO_moveArm:
                self.moveArm(d[3:])

            if d[0] != -10:
                self.visitWaypoint(d[:3])

            if DO_THETA_CORRECTION and d[2] != -10:
                self.rotateToTheta(d[2])

            trans_stretch = self.findPose(STRETCH_FRAME)
            theta = findTheta(trans_stretch, d)
            found_transform = False
            trans_stretch = self.findPose(STRETCH_FRAME)

            rospy.loginfo("Robot is at: x: {:.3f}, y: {:.3f}, z: {:.3f}, theta: {:.3f}".format(trans_stretch.translation.x, trans_stretch.translation.y, trans_stretch.translation.z, theta))

        rospy.loginfo("Completed followTrajectory")

        if IS_SIM and DO_moveArm:
            self.move_group_arm.stop()

    def rotateToTheta(self, arg_goal_theta, arg_close_enough=0.01):
        make_theta_correction = True
        while make_theta_correction:
            trans_stretch = self.findPose(STRETCH_FRAME)
            theta = findTheta(trans_stretch, d)
            cmd_w = arg_goal_theta - theta
            if cmd_w > np.pi:
                cmd_w -= 2 * np.pi
            if cmd_w < -np.pi:
                cmd_w += 2 * np.pi
            if np.abs(cmd_w) < arg_close_enough:
                make_theta_correction = False
            else:
                # rospy.loginfo("cmd_v: {} cmd_w: {}".format(0, cmd_w))
                vel_msg = Twist()
                vel_msg.angular.z = cmd_w
                self.vel_pub.publish(vel_msg)
                self.rate.sleep()

        return True

    def visitWaypoint(self, waypoint_xytheta, arg_close_enough=0.01, arg_epsilon=0.1, arg_maxV=2, arg_wheel2center=0.1778):
        rospy.loginfo("Moving base to: x: {:.2f}, y: {:.2f}, theta: {:.2f}".format(waypoint_xytheta[0], waypoint_xytheta[1], waypoint_xytheta[2]))
        at_waypoint = False
        while not at_waypoint:
            trans_stretch = self.findPose(STRETCH_FRAME)

            dist_to_waypoint = np.sqrt([np.square(waypoint_xytheta[0] - trans_stretch.translation.x) +
                                        np.square(waypoint_xytheta[1] - trans_stretch.translation.y)])
            # rospy.loginfo("Distance to waypoint: {}".format(dist_to_waypoint))

            if dist_to_waypoint < arg_close_enough:
                at_waypoint = True

            if not at_waypoint:
                cmd_vx, cmd_vy, theta = findCommands(trans_stretch, d)
                cmd_v, cmd_w = feedbackLin(cmd_vx, cmd_vy, theta, epsilon)
                # cmd_v, cmd_w = thresholdVel(cmd_v, cmd_w, maxV, wheel2center)
                # _, cmd_v, cmd_w = calc_control_command(cmd_vx, cmd_vy, theta, epsilon)

                # rospy.loginfo("Robot is at: x: {:.3f}, y: {:.3f}, z: {:.3f}, theta: {:.3f}".format(trans_stretch.translation.x, trans_stretch.translation.y, trans_stretch.translation.z, theta))
                # rospy.loginfo("cmd_v: {} cmd_w: {}".format(cmd_v, cmd_w))

                vel_msg = Twist()
                vel_msg.linear.x = cmd_v
                vel_msg.angular.z = cmd_w
                self.vel_pub.publish(vel_msg)
                self.rate.sleep()

        return True

    def moveArm(self, arg_desired_ext_lift_yaw):
        dl = arg_desired_ext_lift_yaw.tolist()
        rospy.loginfo("Moving gripper and arm to: extension: {:.2f}, lift: {:.2f}, rotation: {:.2f}".format(arg_desired_ext_lift_yaw[0], arg_desired_ext_lift_yaw[1], arg_desired_ext_lift_yaw[2]))
        if not IS_SIM:
            with self.move_lock:
                pose = dict()
                if arg_desired_ext_lift_yaw[0] != -10:
                    pose['wrist_extension'] = arg_desired_ext_lift_yaw[0]
                if arg_desired_ext_lift_yaw[1] != -10:
                    pose['joint_lift'] = arg_desired_ext_lift_yaw[1]
                if arg_desired_ext_lift_yaw[2] != -10:
                    pose['joint_wrist_yaw'] = arg_desired_ext_lift_yaw[2]
                # 'rotate_mobile_base': 0.1
                # 'translate_mobile_base': -0.2

                # The base can also be controlled with odometery?
                # translate_mobile_base
                # rotate_mobile_base
                self.move_to_pose(pose)
        else:
            joint_goal = self.move_group_arm.get_current_joint_values()
            if dl[1] != -10:
                if dl[1] < 0:
                    dl[1] = 0
                if dl[1] > 1:
                    dl[1] = 1
                joint_goal[0] = dl[1]
            if dl[0] != -10:
                if dl[0] < 0:
                    dl[0] = 0
                if dl[0] > 0.5:
                    dl[0] = 0.5
                joint_goal[1] = dl[0]/4
                joint_goal[2] = dl[0]/4
                joint_goal[3] = dl[0]/4
                joint_goal[4] = dl[0]/4
            if dl[2] != -10:
                joint_goal[5] = dl[2]
            # rospy.loginfo("Joint goal {}".format(joint_goal))
            self.move_group_arm.go(joint_goal, wait=True)

        return True

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

def findCommands(arg_cur_pose, arg_desired_pose):
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

def findTheta(arg_cur_pose):
    q1 = arg_cur_pose.rotation.x
    q2 = arg_cur_pose.rotation.y
    q3 = arg_cur_pose.rotation.z
    q0 = arg_cur_pose.rotation.w
    theta = np.arctan2(2 * (q1 * q2 + q0 * q3), q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 **2)
    if not IS_SIM:
        theta -= np.pi
    return theta


def findTrajectoryFromDMP(start_pose, end_pose, skill_name, dmp_folder, opts):
    model = DMPNN(opts['start_dimension'], 1024, opts['dimension'], opts['basis_fs']).to(DEVICE)
    model.load_state_dict(torch.load(dmp_folder + skill_name + ".pt"))
    starts = np.zeros([1, 2, start_pose.size])
    starts[0, 0, :] = start_pose
    starts[0, 1, :] = end_pose
    rospy.loginfo("Starts {}".format(starts))
    learned_weights = model(np_to_pgpu(starts))
    dmp = DMP(opts['basis_fs'], opts['dt'], opts['dimension'])
    # print("Calculating rollout")
    learned_rollouts, _, _ = \
        dmp.rollout_torch(torch.tensor(starts[:, 0, :]).to(DEVICE), torch.tensor(starts[:, 1, :]).to(DEVICE), learned_weights)
    out = learned_rollouts[0, :, :].cpu().detach().numpy()

    out = np.vstack([out, end_pose])
    rospy.loginfo("Learned rollout {}".format(out))

    return out


def findObjectPickupPose(obj_pose, obj_name):
    stretch_pose = -10 * np.ones([1, 6])
    if obj_name == "box_1::base_link":
        stretch_pose[0] = obj_pose.translation.x - 0.04
        stretch_pose[1] = obj_pose.translation.y - 0.75
        stretch_pose[2] = np.pi
    elif obj_name == "box_2::base_link":
        stretch_pose[0] = obj_pose.translation.x - 0.75
        stretch_pose[1] = obj_pose.translation.y - 0.04
        stretch_pose[2] = 3 * np.pi / 2
    elif obj_name == "box_3::base_link":
        stretch_pose[0] = obj_pose.translation.x + 0.04
        stretch_pose[1] = obj_pose.translation.y + 0.75
        stretch_pose[2] = 0
    elif obj_name == 'duck':
        stretch_pose[0] = obj_pose.translation.x - 0.04
        stretch_pose[1] = obj_pose.translation.y - 0.75
        stretch_pose[2] = np.pi

    return stretch_pose


def addJointValuesToPose(arg_stretch_pose, arg_joint_values):
    arg_stretch_pose[3] = arg_joint_values[0]
    arg_stretch_pose[4] = arg_joint_values[1]
    arg_stretch_pose[5] = arg_joint_values[2]

    return arg_stretch_pose


if __name__ == '__main__':
    # Extension, lift, yaw
    dmp_opts = json_load_wrapper("/home/adam/repos/synthesis_based_repair/data/stretch/stretch_dmp_opts.json")
    folder_dmps = "/home/adam/repos/synthesis_based_repair/data/dmps/"
    try:
        parser = ap.ArgumentParser(description='Handover an object.')
        args, unknown = parser.parse_known_args()
        node = StretchSkill()

        # Find where the robot is, where it should pickup the block, and how it should get between them
        if IS_SIM:
            # unit_box_pose = node.findPose(OBJ_NAME)
            # base_link = node.findPose(STRETCH_FRAME)
            # start_pose = np.array([base_link.translation.x, base_link.translation.y, 0, -10, -10, -10])
            # end_pose = findObjectPickupPose(unit_box_pose, OBJ_NAME)
            # jv = node.getJointValues()
            # start_pose = addJointValuesToPose(start_pose, jv)
            # end_pose = addJointValuesToPose(end_pose, jv)
            #
            # rospy.loginfo("Startpose: {}".format(start_pose))
            # rospy.loginfo("End pose: {}".format(end_pose))
            #
            # node.openGripper()
            # rospy.loginfo("Moving to box")
            # stretch_base_traj = findTrajectoryFromDMP(start_pose, end_pose, 'skillStretch1to2', folder_dmps, dmp_opts)
            # node.followTrajectory(stretch_base_traj)
            #
            # # correct final pose
            # node.rotateToTheta(end_pose[2])

            rospy.loginfo("Retracting arm")
            stretch_arm_retract = np.array([0, -10, -10])
            node.moveArm(stretch_arm_retract)

            unit_box_pose = node.findPose(OBJ_NAME)
            # TODO: Change the 0.03 to the correct value/find it from the robot URDF
            amount_to_lift = (unit_box_pose.translation.z)-.03
            rospy.loginfo("Lifting arm to: {}".format(amount_to_lift))
            stretch_arm_raise = np.array([0, amount_to_lift, 0])
            node.moveArm(stretch_arm_raise)
            node.openGripper()


            base_link = node.findPose(STRETCH_FRAME)
            # Reduce extension by the default gripper extension (0.34) and the offset of the gazebo box (0.04)
            # TODO check these values on real robot/find from URDF
            amount_to_extend = (unit_box_pose.translation.y - base_link.translation.y) - (0.34 + 0.02)
            rospy.loginfo("Extending arm to: {}".format(amount_to_extend))
            stretch_extend = np.array([ amount_to_extend, -10, -10])
            node.moveArm(stretch_extend)
            # ee_left = node.findPose('robot::link_gripper_finger_left')
            # rospy.loginfo("Stretch left finger after: {}".format(ee_left))

            # node.closeGripper()
            node.attachObject('box_1')

            rospy.loginfo("Lifting arm")
            stretch_lift_with_duck = np.array([-10, amount_to_lift + 0.05, -10])
            node.moveArm(stretch_lift_with_duck)
            stretch_retract_with_duck = np.array([0, -10, -10])
            node.moveArm(stretch_retract_with_duck)

            node.detachObject('box_1')


        if not IS_SIM:
            # Retracts arm all the way, lifts, extends, and lifts a little more

            rospy.loginfo("Retracting arm")
            stretch_arm_retract = np.array([0, -10, -10])
            node.moveArm(stretch_arm_retract)

            unit_box_pose = node.findPose(OBJ_NAME)
            # TODO: Change the 0.03 to the correct value/find it from the robot URDF
            amount_to_lift = (unit_box_pose.translation.z)-.03
            rospy.loginfo("Lifting arm to: {}".format(amount_to_lift))
            stretch_arm_raise = np.array([0, amount_to_lift, 0])
            node.moveArm(stretch_arm_raise)


            base_link = node.findPose(STRETCH_FRAME)
            # Reduce extension by the default gripper extension (0.34) and the offset of the gazebo box (0.04)
            # TODO check these values on real robot/find from URDF
            amount_to_extend = (unit_box_pose.translation.y - base_link.translation.y) - (0.34 + 0.02)
            rospy.loginfo("Extending arm to: {}".format(amount_to_extend))
            stretch_extend = np.array([ amount_to_extend, -10, -10])
            node.moveArm(stretch_extend)
            # ee_left = node.findPose('robot::link_gripper_finger_left')
            # rospy.loginfo("Stretch left finger after: {}".format(ee_left))

            node.closeGripper()

            rospy.loginfo("Lifting arm")
            stretch_lift_with_duck = np.array([-10, amount_to_lift + 0.07, -10])
            node.moveArm(stretch_lift_with_duck)

    except KeyboardInterrupt:
        rospy.loginfo('interrupt received, so shutting down')

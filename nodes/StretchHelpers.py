#!/usr/bin/env python

"""
This file contains helper functions for the stretch robot control.

The helper functions include finding corrections to the robot arm when it
reaches, the theta given the quaternion, etc


"""

from geometry_msgs.msg import Transform
import numpy as np
from math import dist
from tf.transformations import euler_from_quaternion, quaternion_from_euler

def feedbackLin(arg_cmd_vx, arg_cmd_vy, arg_theta, arg_epsilon):
    """ Performs feedback linearization
    """
    cmd_vi = np.array([[arg_cmd_vx], [arg_cmd_vy]])
    R_b_i = np.array([[np.cos(arg_theta), np.sin(arg_theta)], [-np.sin(arg_theta), np.cos(arg_theta)]])
    cmd_vw = np.dot(np.dot(np.array([[1, 0], [0, 1/arg_epsilon]]), R_b_i), cmd_vi)
    cmd_v = cmd_vw[0]
    cmd_w = cmd_vw[1]

    return cmd_v, cmd_w

def thresholdVel(arg_cmd_v, arg_cmd_w, arg_maxV, arg_wheel2center):
    """ Thresholds the velocity of the robot
    """
    maxW = arg_maxV/arg_wheel2center
    ratioV = np.abs(arg_cmd_v/arg_maxV)
    ratioW = np.abs(arg_cmd_w/maxW)
    ratioTot = ratioV + ratioW
    if ratioTot > 1:
        arg_cmd_v = arg_cmd_v/ratioTot
        arg_cmd_w = arg_cmd_w/ratioTot

    return arg_cmd_v, arg_cmd_w

def findCommands(arg_cur_pose, arg_desired_pose):
    """ Finds the x and y velocity commands for the robot
    """
    cmd_vx = arg_desired_pose[0] - arg_cur_pose.translation.x
    cmd_vy = arg_desired_pose[1] - arg_cur_pose.translation.y
    theta = findTheta(arg_cur_pose)
    return cmd_vx, cmd_vy, theta

def findTheta(arg_cur_pose):
    """Finds the orientation of the robot given the pose with quaternion info
    Given the pose of the robot as a transfrom, returns the orientation aka
    theta of the robot. Uses the formula here:
    https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    Args:
        arg_cur_pose: transform_msg
    Returns:
        theta: np.array
    """
    q1 = arg_cur_pose.rotation.x
    q2 = arg_cur_pose.rotation.y
    q3 = arg_cur_pose.rotation.z
    q0 = arg_cur_pose.rotation.w
    # theta = np.arctan2(2 * (q1 * q2 + q0 * q3), q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 **2)
    # theta = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    (_, _, theta) = euler_from_quaternion([q1, q2, q3, q0])
    # if not IS_SIM:
    #     theta -= np.pi
    while theta < -np.pi/4 or theta >= 2*np.pi-np.pi/4:
        if theta < -np.pi/4:
            theta += 2*np.pi
        if theta >= 2*np.pi-np.pi/4:
            theta -= 2*np.pi
    return theta


def findArmExtensionAndRotation(goal_pose, robot_pose):
    """ Finds the amount to extend the arm and rotate the wrist to reach a goal point
    """
    GtoW = 0.23 #Grip to Wrist Distance

    xd = goal_pose.translation.x
    yd = goal_pose.translation.y

    xr = robot_pose.translation.x
    yr = robot_pose.translation.y

    qrobot = findTheta(robot_pose)
    print("ThetaRobot -> ", qrobot)
    qr = qrobot - (np.pi)/2
    print("thetaARM -> ",qr)

    p = yr - (xr*np.tan(qr)) - yd

    #Solving quadratic equation

    a = 1 + (np.tan(qr)**2)
    b = (-2*xd) + (2*p*np.tan(qr))
    c = xd**2 + p**2 - GtoW**2

    # Discriminant

    d = (b**2) - (4*a*c)
    # X values

    x1 = (-b + np.sqrt(d))/(2*a)
    x2 = (-b - np.sqrt(d))/(2*a)

    y1 = (np.tan(qr) * x1) + yr - (xr*np.tan(qr))
    y2 = (np.tan(qr) * x2) + yr - (xr*np.tan(qr))

    pt1 = (x1,y1)
    pt2 = (x2,y2)

    robot = (xr,yr)

    dist1 = dist(pt1,robot)
    dist2 = dist(pt2,robot)

    if dist1<dist2:
        amount_to_extend = dist1

    else:
         amount_to_extend = dist2

    print("AmountEXTEND -> ", amount_to_extend)
    g = (yd - yr - (amount_to_extend * np.sin(qr)))/GtoW
    print("Value of g -> ", g)
    print("sin inverse ->",np.arcsin(g))
    wrist_theta = np.arcsin(g) - qr
    print("ThetaWrist -> ", wrist_theta)

    # amount_to_extend = 0
    # wrist_theta = 0

    return amount_to_extend, wrist_theta


def forwardKinematicsStretch(robot_pose, arm_extension, theta_wrist):
    """ Finds the forward kinematics of the stretch (in 2d) given the pose

    Args:
        robot_pose: Transform
        arm_extension: double
        theta_wrist: double (radians)

    Returns:
        ee_pose: Transform
    """

    l_wrist = 0.23
    x_robot = robot_pose.translation.x
    y_robot = robot_pose.translation.y
    t_robot = findTheta(robot_pose)
    t_wrist = theta_wrist
    x_ee = arm_extension * np.cos(t_robot - np.pi/2) + l_wrist * np.cos(t_robot + theta_wrist - np.pi/2) + x_robot
    y_ee = arm_extension * np.sin(t_robot - np.pi/2) + l_wrist * np.sin(t_robot + theta_wrist - np.pi/2) + y_robot

    ee_pose = Transform()
    ee_pose.translation.x = x_ee
    ee_pose.translation.y = y_ee

    return ee_pose


def testArmExtensionAndRotation():
    """ Verifies that the arm extension and wrist rotation calculations are correct
    """
    robot_pose = Transform()
    robot_pose.rotation.w = 1
    extension = 0.5441
    theta_wrist = 1

    ee = forwardKinematicsStretch(robot_pose, extension, theta_wrist)
    print("EE x: {} EE y: {}".format(ee.translation.x, ee.translation.y))
    found_extension, found_theta = findArmExtensionAndRotation(ee, robot_pose)

    print("Found ext: {} Found theta: {}".format(found_extension, found_theta))
    print("Inp ext: {} Inp theta: {}".format(extension, theta_wrist))


if __name__ == "__main__":
    testArmExtensionAndRotation()

#!/usr/bin/env python

"""
This file contains helper functions for the stretch robot control.

The helper functions include finding corrections to the robot arm when it
reaches, the theta given the quaternion, etc


"""

from geometry_msgs.msg import Transform
import numpy as np
from math import dist

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
    q1 = arg_cur_pose.rotation.x
    q2 = arg_cur_pose.rotation.y
    q3 = arg_cur_pose.rotation.z
    q0 = arg_cur_pose.rotation.w
    theta = np.arctan2(2 * (q1 * q2 + q0 * q3), q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 **2)
    if not IS_SIM:
        theta -= np.pi
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
    theta = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    # if not IS_SIM:
        # theta -= np.pi
    return theta


def eulerToQuaternion(yaw, pitch, roll):
    """Given the yaw, pitch, and roll, finds the transform

    Note: this returns the full transform, including the translation, which is 0
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    q = Transform()
    q.rotation.w = cr * cp * cy + sr * sp * sy
    q.rotation.x = sr * cp * cy - cr * sp * sy
    q.rotation.y = cr * sp * cy + sr * cp * sy
    q.rotation.z = cr * cp * sy - sr * sp * cy

    return q


def findArmExtensionAndRotation(goal_pose, robot_pose_x, robot_pose_y, robot_theta):
    """ Finds the amount to extend the arm and rotate the wrist to reach a goal point
    """
    GtoW = 0.23 #Grip to Wrist Distance

    xd = goal_pose.translation.x
    yd = goal_pose.translation.y
    xr = robot_pose_x
    yr = robot_pose_y

    # if robot_theta < -0.1:
    #     qrobot = robot_theta + 2*np.pi
    # elif robot_theta >= 2*np.pi:
    #     qrobot = robot_theta - (2*np.pi)

    qrobot = robot_theta
    q_arm = qrobot - (np.pi)/2 #Robot arm theta

    p = yr - (xr*np.tan(q_arm)) - yd

    #Solving quadratic equation
    a = 1 + (np.tan(q_arm)**2)
    b = (-2*xd) + (2*p*np.tan(q_arm))
    c = xd**2 + p**2 - GtoW**2

    # Discriminant
    d = (b**2) - (4*a*c)

    # X values
    x1 = (-b + np.sqrt(d))/(2*a)
    x2 = (-b - np.sqrt(d))/(2*a)

    y1 = (np.tan(q_arm) * x1) + yr - (xr*np.tan(q_arm))
    y2 = (np.tan(q_arm) * x2) + yr - (xr*np.tan(q_arm))

    pt1 = (x1,y1)
    pt2 = (x2,y2)
    robot = (xr,yr)

    dist1 = dist(pt1,robot)
    dist2 = dist(pt2,robot)

    if dist1<dist2:
        amount_to_extend = dist1
    else:
         amount_to_extend = dist2

    if amount_to_extend == dist1:
        x_pt = pt1[0]
        y_pt = pt1[1]
    else:
        x_pt = pt2[0]
        y_pt = pt2[1]

    # print("Point is ->", point)

    # g = (yd - yr - (amount_to_extend * np.sin(q_arm)))/GtoW
    # wrist_theta = np.arcsin(g) - q_arm

    angle = np.arctan2(yd - y_pt, xd - x_pt)
    wrist_theta = angle - q_arm

    if wrist_theta>=2*np.pi:
        wrist_theta = wrist_theta - (2*np.pi)
    elif wrist_theta < -0.1:
        wrist_theta = wrist_theta + (2*np.pi)

    return amount_to_extend, wrist_theta


def forwardKinematicsStretch(robot_posex, robot_posey, robottheta, arm_extension, theta_wrist, l_wrist = 0.23):
    """ Finds the forward kinematics of the stretch (in 2d) given the pose

    Args:
        robot_pose: Transform
        arm_extension: double
        theta_wrist: double (radians)

    Returns:
        ee_pose: Transform
    """

    # x_robot = robot_pose.translation.x
    # y_robot = robot_pose.translation.y
    # t_robot = findTheta(robot_pose)

    x_robot = robot_posex
    y_robot = robot_posey
    t_robot = robottheta

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
    # robot_pose = Transform()
    # robot_pose.translation.x = 0.1
    # robot_pose.translation.y = 0.1
    # robot_pose.rotation.w = 1

    # xr_range = np.linspace(-2,2,20)
    # yr_range = np.linspace(-2,2,20)
    # angles = np.linspace(-2 * np.pi, 2 * np.pi,20)
    # extension =np.linspace(0,1,num=20, endpoint = False)
    # theta_wrist = np.linspace(0, np.pi/2, num=20, endpoint = False)

    # theta_wrist_b = np.linspace(np.pi/2+0.5, 3*np.pi/2, num=500, endpoint=False)[1:]
    # theta_wrist = np.hstack((theta_wrist_a, theta_wrist_b))

    xr_range = [1]
    yr_range = [1]
    angles = [0.0001]
    extension = [0]
    theta_wrist = [0.8]


    #LOOPING
    for x in xr_range:
        for y in yr_range:
            for ang in angles:
                for ex in extension:
                    for tw in theta_wrist:
                        ee = forwardKinematicsStretch(x, y, ang, ex, tw)
                        # print("Robot x: {} Robot y: {}".format(robot_pose.translation.x, robot_pose.translation.y))

                        # print("Robot x: {} Robot y: {}".format(x, y))
                        # print("EE x: {} EE y: {}".format(ee.translation.x, ee.translation.y))
                        found_extension, found_theta = findArmExtensionAndRotation(ee, x, y, ang)

                        ee_found = forwardKinematicsStretch(x, y, ang, found_extension, found_theta)

                        # print("Found ext: {} Found theta: {}".format(found_extension, found_theta))
                    # print("Inp ext: {} Inp theta: {}".format(extension, theta_wrist))

                        errorExt = abs(ex - found_extension)
                        errorTheta = abs(tw - found_theta)
                        ee_x_diff = ee_found.translation.x - ee.translation.x
                        ee_y_diff = ee_found.translation.y - ee.translation.y

                        if True: #((errorExt > 0.001) or (errorTheta > 0.001)):
                        # print("X: {:5.5f} Y: {:5.5f} Qr: {:5.5f} Ext: {:5.5f} Qw: {:5.5f}    =>     ErrorExtension: {:5.5f}          ErrorWrist: {:5.5f}".format(robot_pose.translation.x, robot_pose.translation.y, robot_pose.rotation.w, ex, tw, errorExt, errorTheta))

                            print("X: {:5.5f} Y: {:5.5f} Qr: {:5.5f} Ext: {:5.5f} Qw: {:5.5f} => \t ErrorExtension: {:5.5f} \t ErrorWrist: {:5.5f} \t EE_x_desired: {:5.5f} \t EE_x_found: {:5.5f} \t EE_y_desired: {:5.5f} \t EE_y_found: {:5.5f}".format(x, y, ang, ex, tw, errorExt, errorTheta, ee.translation.x, ee_found.translation.x, ee.translation.y, ee_found.translation.y))
                            # print("X: {:5.5f} Y: {:5.5f} Qr: {:5.5f} Ext: {:5.5f} Qw: {:5.5f} => \t ErrorExtension: {:5.5f} \t ErrorWrist: {:5.5f} \t EE_x_diff: {:5.5f} \t EE_y_diff: {:5.5f}".format(x, y, ang, ex, tw, errorExt, errorTheta, ee_x_diff, ee_y_diff))

                        # print("-"*190)
                        # print("-"*190)
                    # elif (errorExt == 'nan') or (errorTheta == 'nan'):
                        # print("X: {:5.5f} Y: {:5.5f} Qr: {:5.5f} Ext: {:5.5f} Qw: {:5.5f}    =>     ErrorExtension: {:5.5f}          ErrorWrist: {:5.5f}".format(robot_pose.translation.x, robot_pose.translation.y, robot_pose.rotation.w, ex, tw, errorExt, errorTheta))


if __name__ == "__main__":
    testArmExtensionAndRotation()

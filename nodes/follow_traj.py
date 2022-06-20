#!/usr/bin/env python

from copy import deepcopy
import rospy
import pickle
import tf2_ros
import numpy as np
import copy
from geometry_msgs.msg import Pose, Point, TransformStamped, Twist, Transform, PoseWithCovarianceStamped
import matplotlib.pyplot as plt
from std_msgs.msg import Bool

ORIGIN_FRAME = 'base_link'
STRETCH_FRAME = 'odom'
CMD_VEL_TOPIC = '/stretch_diff_drive_controller/cmd_vel'

# ORIGIN_FRAME = 'origin'
# STRETCH_FRAME = 'stretch'
# CMD_VEL_TOPIC = '/stretch/cmd_vel'

def main(file_aut, file_structured_slugs, file_symbols):
    rospy.init_node('run_strategy_node')
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    vel_pub = rospy.Publisher(CMD_VEL_TOPIC, Twist, queue_size=10)
    rate = rospy.Rate(10.0)


    while not rospy.is_shutdown():
        try:
            trans_stretch_stamped = tfBuffer.lookup_transform(ORIGIN_FRAME, STRETCH_FRAME, rospy.Time.now(), rospy.Duration(2.0))
            trans_stretch = trans_stretch_stamped.transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rate.sleep()
            print("Waiting for transform")
            continue

        q1 = trans_stretch.rotation.x
        q2 = trans_stretch.rotation.y
        q3 = trans_stretch.rotation.z
        q0 = trans_stretch.rotation.w
        # theta = np.arctan2(2 * (q1 * q2 + q0 * q3), q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 ** 2)
        theta = np.arctan2(2 * (q1 * q2 + q0 * q3), q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 ** 2)# - np.pi
        # if theta < 0:
        #     theta += 2 * np.pi

        print("Robot is at: x: {:.3f}, y: {:.3f}, z: {:.3f}, theta: {:.3f}".format(trans_stretch.translation.x, trans_stretch.translation.y, trans_stretch.translation.z, theta))

        wp_array = np.array([[0, 0], [0, 0], [1, 1]])
        n_rollouts = len(wp_array)

        # plot resp.out_pose
        # use feedback linearization to drive to the waypoints
        close_enough = 0.1
        gotopt = 1
        waypoint = wp_array[gotopt]
        epsilon = 0.1
        maxV = .2
        wheel2center = 0.1778

        while gotopt < n_rollouts:
            try:
                trans_stretch_stamped = tfBuffer.lookup_transform(ORIGIN_FRAME, STRETCH_FRAME, rospy.Time.now(), rospy.Duration(2.0))
                # trans_stretch_stamped = tfBuffer.lookup_transform('odom', 'base_link', rospy.Time.now(), rospy.Duration(2.0))
                trans_stretch = trans_stretch_stamped.transform
                print("Robot is at: x: {:.3f}, y: {:.3f}, z: {:.3f}, theta: {:.3f}".format(trans_stretch.translation.x, trans_stretch.translation.y, trans_stretch.translation.z, theta))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rate.sleep()
                continue

            # Find commands to go to waypoint
            dist_to_waypoint = np.sqrt([np.square(waypoint[0] - trans_stretch.translation.x) +
                                        np.square(waypoint[1] - trans_stretch.translation.y)])

            if dist_to_waypoint < close_enough:
                gotopt += 1
                if gotopt < n_rollouts:
                    waypoint = wp_array[gotopt]

            if gotopt < n_rollouts:
                cmd_vx = waypoint[0] - trans_stretch.translation.x
                cmd_vy = waypoint[1] - trans_stretch.translation.y
                q1 = trans_stretch.rotation.x
                q2 = trans_stretch.rotation.y
                q3 = trans_stretch.rotation.z
                q0 = trans_stretch.rotation.w
                theta = np.arctan2(2 * (q1 * q2 + q0 * q3), q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 **2) - np.pi
                # if theta < 0:
                #     theta += 2 * np.pi
                # print("cmdX: {} cmdY: {} theta: {}".format(cmd_vx, cmd_vy, theta))

                cmd_v, cmd_w = feedbackLin(cmd_vx, cmd_vy, theta, epsilon)
                # _, cmd_v, cmd_w = calc_control_command(cmd_vx, cmd_vy, theta, epsilon)

                print("cmd_v: {} cmd_w: {}".format(cmd_v, cmd_w))

                maxW = maxV/wheel2center
                ratioV = np.abs(cmd_v/maxV)
                ratioW = np.abs(cmd_w/maxW)
                ratioTot = ratioV + ratioW
                if ratioTot > 1:
                    cmd_v = cmd_v/ratioTot
                    cmd_w = cmd_w/ratioTot

                vel_msg = Twist()
                vel_msg.linear.x = cmd_v[0]
                vel_msg.angular.z = cmd_w[0]
                # print("fb lin cmd_v: {} cmd_w: {}".format(cmd_v, cmd_w))
                print(vel_msg)
                vel_pub.publish(vel_msg)
                rate.sleep()

        print("Finished execution")


def feedbackLin(arg_cmd_vx, arg_cmd_vy, arg_theta, arg_epsilon):
    cmd_vi = np.array([[arg_cmd_vx], [arg_cmd_vy]])
    R_b_i = np.array([[np.cos(arg_theta), np.sin(arg_theta)], [-np.sin(arg_theta), np.cos(arg_theta)]])
    cmd_vw = np.dot(np.dot(np.array([[1, 0], [0, 1/arg_epsilon]]), R_b_i), cmd_vi)
    cmd_v = cmd_vw[0]
    cmd_w = cmd_vw[1]

    return cmd_v, cmd_w

def calc_control_command(x_diff, y_diff, theta, theta_goal):
    """
    Returns the control command for the linear and angular velocities as
    well as the distance to goal
    Parameters
    ----------
    x_diff : The position of target with respect to current robot position
             in x direction
    y_diff : The position of target with respect to current robot position
             in y direction
    theta : The current heading angle of robot with respect to x axis
    theta_goal: The target angle of robot with respect to x axis
    Returns
    -------
    rho : The distance between the robot and the goal position
    v : Command linear velocity
    w : Command angular velocity
    """

    # Description of local variables:
    # - alpha is the angle to the goal relative to the heading of the robot
    # - beta is the angle between the robot's position and the goal
    #   position plus the goal angle
    # - Kp_rho*rho and Kp_alpha*alpha drive the robot along a line towards
    #   the goal
    # - Kp_beta*beta rotates the line so that it is parallel to the goal
    #   angle
    #
    # Note:
    # we restrict alpha and beta (angle differences) to the range
    # [-pi, pi] to prevent unstable behavior e.g. difference going
    # from 0 rad to 2*pi rad with slight turn
    Kp_rho = 5
    Kp_alpha = 8
    Kp_beta = 2
    rho = np.hypot(x_diff, y_diff)
    alpha = (np.arctan2(y_diff, x_diff)
             - theta + np.pi) % (2 * np.pi) - np.pi
    beta = (theta_goal - theta - alpha + np.pi) % (2 * np.pi) - np.pi
    v = Kp_rho * rho
    w = Kp_alpha * alpha - Kp_beta * beta

    if alpha > np.pi / 2 or alpha < -np.pi / 2:
        v = -v

    return rho, v, w



if __name__ == "__main__":
    rospy.sleep(10)
    file_symbols = "/home/adam/catkin_ws/src/hscc2022/data/symbols/find_symbols.json"
    file_skills = "/home/adam/catkin_ws/src/hscc2022/data/trajectories/find_skills_new.pkl"
    file_structured_slugs = "/home/adam/catkin_ws/src/hscc2022/data/specifications/k_find.structuredslugs"
    file_aut = "/home/adam/catkin_ws/src/hscc2022/data/specifications/k_find_strategy.aut"
    main(file_aut, file_structured_slugs, file_symbols)

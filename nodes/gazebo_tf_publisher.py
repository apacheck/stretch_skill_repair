#!/usr/bin/env python
import rospy
import tf2_ros
import gazebo_msgs.msg
import geometry_msgs.msg
import time

import pdb

IS_SIM = True

if IS_SIM:
    ORIGIN_FRAME = 'odom'
else:
    ORIGIN_FRAME = 'origin'

if __name__ == '__main__':
    rospy.init_node('gazebo_tf_broadcaster')

    broadcaster = tf2_ros.StaticTransformBroadcaster()

    publish_frequency = rospy.get_param("publish_frequency", 10)

    last_published = None

    def callback(data):
        global last_published
        if last_published and publish_frequency > 0.0 and time.time() - last_published <= 1 / publish_frequency:
            return
        transforms = []
        for i in range(len(data.name)):
            # if data.name[i] == "robot::base_link":
            #     child_frame = 'base_link'
            # else:
            #     continue
            transform = geometry_msgs.msg.TransformStamped()
            transform.header.stamp = rospy.Time.now()
            transform.header.frame_id = ORIGIN_FRAME
            transform.child_frame_id = data.name[i]
            # transform.child_frame_id = child_frame
            transform.transform.translation.x = data.pose[i].position.x
            transform.transform.translation.y = data.pose[i].position.y
            transform.transform.translation.z = data.pose[i].position.z
            transform.transform.rotation.w = data.pose[i].orientation.w
            transform.transform.rotation.x = data.pose[i].orientation.x
            transform.transform.rotation.y = data.pose[i].orientation.y
            transform.transform.rotation.z = data.pose[i].orientation.z
            transforms.append(transform)
            # rospy.loginfo("Publishing: {} as {}".format(data.name[i], child_frame))
        broadcaster.sendTransform(transforms)
        last_published = time.time()

    rospy.Subscriber("/gazebo/link_states", gazebo_msgs.msg.LinkStates, callback)

    rospy.spin()

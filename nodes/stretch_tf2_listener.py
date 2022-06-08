#!/usr/bin/env python
import rospy

import math
import tf2_ros
import geometry_msgs.msg
from sensor_msgs.msg import JointState

def callback(data):
    print('here')
    # rospy.loginfo(rospy.get_caller_id() + "I heard %s", data)
    fid = open('/home/adam/Documents/tmp_stretch.txt', 'a')
    fid.write("{}\n".format(data.position))
    fid.close()

def main():
    rospy.init_node('stretch_tf2_listener')

    # tfBuffer = tf2_ros.Buffer()
    # listener = tf2_ros.TransformListener(tfBuffer)

    stretch_joints = rospy.Subscriber("/stretch/joint_states", JointState, callback)
    # turtle_vel = rospy.Publisher('%s/cmd_vel' % turtle_name, geometry_msgs.msg.Twist, queue_size=1)

    # rate = rospy.Rate(10.0)
    # while not rospy.is_shutdown():
    #     try:
    #         trans = tfBuffer.lookup_transform('world', 'stretch/base_link', rospy.Time())
    #         print(trans)
    #     except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
    #         rate.sleep()
    #         continue


        # msg = geometry_msgs.msg.Twist()
        #
        # msg.angular.z = 4 * math.atan2(trans.transform.translation.y, trans.transform.translation.x)
        # msg.linear.x = 0.5 * math.sqrt(trans.transform.translation.x ** 2 + trans.transform.translation.y ** 2)
        #
        # turtle_vel.publish(msg)

        # rate.sleep()
    rospy.spin()


if __name__ == '__main__':
    main()

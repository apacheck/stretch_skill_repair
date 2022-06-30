# stretch_skill_repair

Code for running skills on the Stretch robot.
For use with synthesis_based_repair

# Setting up the stretch

Turn on the stretch.
Either ssh or connect directly to the stretch and run:
```shell
roslaunch stretch_core stretch_driver.launch
roslaunch stretch_skill_repair mocap_stretch.launch
```

# In simulation
We will use https://github.com/pal-robotics/gazebo_ros_link_attacher to make the gripper successfully pick up an object in simulation.
This line: \<plugin name="ros_link_attacher_plugin" filename="libgazebo_ros_link_attacher.so"/\> is added to the world file to enable it.

You may need to run catkin build or catkin_make and make some of the files executable
```shell
roslaunch stretch_skill_repair gazebo.launch world:=[PATH TO CATKIN_WS]/catkin_ws/src/stretch_skill_repair/worlds/three_ducks.world
```

In a separate terminal
```shell
roslaunch stretch_moveit_config demo_gazebo.launch
```

In a separate terminal
```shell
rosrun stretch_skill_repair gazebo_tf_publisher.py
```

In a separate terminal
```shell
rosrun stretch_skill_repair StretchSkill.py
```
# Notes
You may need to install ros-$ROS_DISTRO-realsense2-description and ros-$ROS_DISTRO-moveit

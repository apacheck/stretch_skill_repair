# stretch_skill_repair

Code for running skills on the Stretch robot.
For use with synthesis_based_repair

# Setting up the stretch

Turn on the stretch.
Either ssh or connect directly to the stretch and run:
```shell
roslaunch stretch_core stretch_driver.launch
```

# In simulation
You may need to run catkin build or catkin_make and make some of the files executable
```shell
roslaunch stretch_gazebo gazebo.launch world:=[PATH TO CATKIN_WS]/catkin_ws/src/stretch_skill_repair/worlds/three_tables2.world
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
You may need to install ros-$ROS-DISTRO-realsense2-camera

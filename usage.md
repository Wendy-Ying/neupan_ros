## simulation
```
roslaunch neupan_ros gazebo_limo_env_complex_20.launch
```

## real world
```
sudo ip link set can0 up type can bitrate 500000
source /home/goodboy/scout_ws/devel/setup.bash
roslaunch scout_bringup scout_mini_robot_base.launch
```
```
roslaunch livox_ros_driver2 msg_MID360.launch
rosrun neupan_ros livox_to_laser_scan.py
rosrun neupan_ros livox_tf_broadcaster.py
roslaunch fast_lio mapping_mid360.launch 
```
## run
```
roslaunch neupan_ros neupan_gazebo_limo.launch
```
## simulation
```
roslaunch neupan_ros gazebo_limo_env_complex_20.launch
```

## real world
```
./hanjing_scripts/bringup_scout.sh
roslaunch vrpn_client_ros sample.launch server:=192.168.1.109(192.168.50.104)
```
```
roslaunch neupan_ros run_env_livox.launch
roslaunch fast_lio mapping_mid360.launch
rosrun neupan_ros helmet_velocity.py
```
## run
```
conda activate neupan
roslaunch neupan_ros neupan_gazebo_limo.launch
```
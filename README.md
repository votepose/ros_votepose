# ros_votepose

## Introduction
This repository contains code for data processing and visualization for 6D object pose estimation.

## Installation
The tools require full ROS installation. The installation assumes you have Ubuntu 18.04 LTS [ROS Melodic]
   ```bash
   $ cd ~/catkin_ws
   $ catkin_make install
   ```
## Launch files

### To assign part id to points:
   ```bash
   $ roslaunch ros_votepose launch_assign_part_id.launch
   ```
### To convert depth to point cloud and visualize gt object pose:
   ```bash
   $ roslaunch ros_votepose launch_bop.launch
   ```
### To evaluate predicted poses:
   ```bash
   $ roslaunch ros_votepose launch_bop_eval.launch
   ```
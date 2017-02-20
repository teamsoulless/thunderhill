# Ros

## Installation

[http://wiki.ros.org/ROS/Installation](http://wiki.ros.org/ROS/Installation)

## Rosrepo

It is good to work with rosrepo. Install rosrepo and:

```
cd thunderhill/ros
rosrepo init
#add source line to your ~/.bashrc
source ~/.bashrc
rosrepo include --all
rosrepo build
```

`rosrepo list` output:

```
Package  RepositoryStatus
-------  ----------------
velodyne velodyne-ros  +
velodyne_driver  velodyne-ros  +b
velodyne_msgsvelodyne-ros  +b
velodyne_pointcloud  velodyne-ros  +b

```

## Node-manager FKIE

Useful for complex systems:

```
sudo apt install ros-kinetic-node-manager-fkie
```


## Reading velodyne pcap

There is binary version of VeloView for Win/Mac. It works, it can slowly export data to csv. It can be compiled on Linux (not so easy), but it is not necessary because we have rosnode that reads pcap: [wiki.ros.org/velodyne_pointcloud](http://wiki.ros.org/velodyne_pointcloud)

### Dataset
The   [Velodyne dataset](http://data.selfracingcars.com/thunderhill/velodyne/velodyne.tar.gz) is available [here](http://data.selfracingcars.com/).
Download it and unpack to `your_dir_with_downloaded_data`.

**This is Velodyne packets only, without GPS nor IMU.**

### Requirements

`velodyne_pointcloud` is required. It can be downloaded from [here](https://github.com/ros-drivers/velodyne).
It needs to be compiled (catkin or rosrepo) - it is not available as precompiled software (AFAIK).

### Build

**Rosrepo**
```
rosrepo include velodyne velodyne_driver velodyne_msgs velodyne_pointcloud
rosrepo build
```

```
Finished  <<< velodyne_pointcloud  [ 39.8 seconds ]
[build] Summary: All 4 packages succeeded!
[build]   Ignored:   1 packages were skipped or are blacklisted.
[build]   Warnings:  2 packages succeeded with warnings.
[build]   Abandoned: None.
[build]   Failed:None.
[build] Runtime: 58.5 seconds total.  
[build] Note: Workspace packages have changed, please re-source setup files to use them.
```

`source ~/.bashrc`

**Catkin**

### Run

```
cd your_dir_with_downloaded_data
```

**HDL32**:

```
roslaunch velodyne_pointcloud 32e_points.launch pcap:=$(pwd)/velodyne/2016-05-28\ Heat\ 1\ Velodyne-HDL-32-Data.pcap
```

### Topics

`rostopic list` ouput:

```
/clicked_point
/diagnostics
/initialpose
/move_base_simple/goal
/rosout
/rosout_agg
/tf
/tf_static
/velodyne_nodelet_manager/bond
/velodyne_nodelet_manager_cloud/parameter_descriptions
/velodyne_nodelet_manager_cloud/parameter_updates
/velodyne_nodelet_manager_driver/parameter_descriptions
/velodyne_nodelet_manager_driver/parameter_updates
/velodyne_packets
/velodyne_points
```

`rostopic hz /velodyne_points ` output:

```
subscribed to [/velodyne_points]
average rate: 9.966
```

### Rviz config

Rviz config is stored in `rviz-configs/velodyne-cloud.rviz`.

Intensity colored point cloud data in Rviz:
![Velodyne data](images/velodyne-rziv.png)

**Youtube video:**

[![Youtube video - Velodyne HDL32 data from Thunderhill west](http://img.youtube.com/vi/HHnDS5O7Pd4/0.jpg)](http://www.youtube.com/watch?v=HHnDS5O7Pd4 "Velodyne HDL32 data from Thunderhill west")


# aruco_pose_estimator_cpp

A ROS 2 C++ node for detecting ArUco markers from onboard camera streams, estimating their 3D pose, and publishing the marker pose in the **drone frame** (`drone_base`).

This package is designed for a setup with two RGB cameras:

- `/cam_1/color/image_raw`
- `/cam_2/color/image_raw`

The node detects only the landing markers of interest:

- **ID 1** → `Medical`
- **ID 2** → `Supply`

The marker side length is assumed to be **1.0 m**.

---

## Features

- Subscribes to two camera image topics
- Detects **4x4 ArUco markers**
- Filters detections to **IDs 1 and 2 only**
- Estimates marker pose using camera intrinsics and marker size
- Transforms the pose from the **camera frame** into the **drone frame**
- Publishes:
  - marker pose
  - landing-place type
  - human-readable debug message
- Runs headless by default (no OpenCV GUI window)

---

## Package Overview

The main node:

- `aruco_pose_estimator`

Main source file:

- `src/aruco_pose_estimator.cpp`

---

## Topics

### Subscribed Topics

| Topic | Type | Description |
|---|---|---|
| `/cam_1/color/image_raw` | `sensor_msgs/msg/Image` | Front RGB camera image |
| `/cam_2/color/image_raw` | `sensor_msgs/msg/Image` | Tilted RGB camera image |

### Published Topics

| Topic | Type | Description |
|---|---|---|
| `/aruco/pose` | `geometry_msgs/msg/PoseStamped` | Marker pose expressed in `drone_base` |
| `/aruco/landing_place_type` | `std_msgs/msg/String` | Landing-place type: `Medical` or `Supply` |
| `/aruco/message` | `std_msgs/msg/String` | Debug string with marker ID, type, source camera, and position |

---

## Marker Semantics

This package only uses the following marker IDs:

| Marker ID | Meaning |
|---|---|
| `1` | `Medical` |
| `2` | `Supply` |

All other detected marker IDs are ignored.

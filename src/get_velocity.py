#!/usr/bin/env python
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

def transform_points(points, pose):
    angles = np.linspace(-np.pi, np.pi, len(points))
    valid_mask = points < 9
    valid_points = points[valid_mask]
    valid_angles = angles[valid_mask]
    cartesian_points = np.column_stack((valid_points * np.cos(valid_angles),
                                        valid_points * np.sin(valid_angles)))
    x_pose, y_pose, theta = pose[:3].flatten()
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    if cartesian_points.shape[0] != 0:
        transformed_points = np.dot(cartesian_points, rotation_matrix.T) + [x_pose, y_pose]
    else:
        transformed_points = np.empty((0, 2))
    return transformed_points

def process_lidar_frame(lidar_scan, state, helmet_msg):
    if lidar_scan is None or helmet_msg is None:
        return None, None

    helmet_pos = np.array([helmet_msg.linear.x, helmet_msg.linear.y])
    helmet_vel = np.array([helmet_msg.angular.x, helmet_msg.angular.y])

    transformed_points = transform_points(lidar_scan, state[:3])

    velocities = np.zeros_like(transformed_points)
    radius = 0.3

    if transformed_points.shape[0] > 0:
        dists = np.linalg.norm(transformed_points - helmet_pos, axis=1)
        mask = dists < radius
        velocities[mask] = helmet_vel

        return transformed_points.T, velocities.T

    return None, None
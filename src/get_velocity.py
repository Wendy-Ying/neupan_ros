#!/usr/bin/env python
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.means_ = np.array([[0.0], [0.1], [-0.1]])
gmm.covariances_ = np.array([[[0.002]], [[0.002]], [[0.002]]])
gmm.weights_ = np.array([0.4, 0.3, 0.3])
gmm.precisions_cholesky_ = np.array([[[1/np.sqrt(0.002)]]]*3)

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

def process_lidar_frame(lidar_scan, state, helmet_msg, num_steps=30, step_size=0.04):
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

        # predict velocity transform to points
        predicted_points = []
        predicted_velocities = []

        for i, (pos, vel) in enumerate(zip(transformed_points, velocities)):
            speed = np.linalg.norm(vel)
            if speed > 1e-1:
                direction = vel / speed
                perp = np.array([-direction[1], direction[0]])
                for j in range(1, num_steps + 1):
                    base_point = pos + direction * step_size * j
                    offset = gmm.sample(1)[0].item()
                    curved_point = base_point + perp * offset
                    predicted_points.append(curved_point)
                    predicted_velocities.append(vel)

        if predicted_points:
            predicted_points = np.array(predicted_points)
            predicted_velocities = np.array(predicted_velocities)
            transformed_points = np.vstack([transformed_points, predicted_points])
            velocities = np.vstack([velocities, predicted_velocities])

        return transformed_points.T, velocities.T

    return None, None
import irsim
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN

def transform_points(points, pose):
    """
    Convert LiDAR points to global coordinates based on robot pose.
    points (np.ndarray): (N,)
    pose (np.ndarray): (3, 1)
    transformed_points (np.ndarray): (N, 2)
    """
    angles = np.linspace(-np.pi, np.pi, len(points))
    valid_mask = points < 2
    valid_points = points[valid_mask]
    valid_angles = angles[valid_mask]
    cartesian_points = np.column_stack((valid_points * np.cos(valid_angles), valid_points * np.sin(valid_angles)))

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


def local_motion_estimation_cluster(frames, times, eps=1, min_samples=2, previous_velocity_map=None, alpha=0.7):
    """
    Estimate motion per cluster using DBSCAN on current frame,
    then use nearest neighbor from previous frames to estimate velocity.
    Applies EMA smoothing to velocity vectors.
    Returns: (N, 5): [x, y, speed, angle], and updated velocity map
    """
    # Filter out empty frames
    frames = [f for f in frames if f.shape[0] > 0]
    times = [t for f, t in zip(frames, times) if f.shape[0] > 0]

    if len(frames) < 2:
        if len(frames) == 0:
            return np.empty((0, 4)), {}
        ref_frame = frames[-1]
        return np.hstack((ref_frame, np.zeros((len(ref_frame), 1)), np.zeros((len(ref_frame), 1)))), {}

    ref_frame = frames[-1]
    total_dt = times[-1] - times[0]
    if ref_frame.shape[0] == 0:
        return np.empty((0, 4)), {}

    cluster_labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(ref_frame)

    # Calculate centroids for current frame clusters
    centroids_now = {
        label: ref_frame[cluster_labels == label].mean(axis=0)
        for label in set(cluster_labels) if label != -1
    }

    # Estimate displacement by matching with previous frames
    velocity_map = {}
    for label, center_now in centroids_now.items():
        disps = []
        for f in frames[:-1]:
            if f.shape[0] == 0:
                continue
            tree = KDTree(f)
            _, idx = tree.query(center_now)
            if f.shape[0] > idx:
                center_prev = f[idx]
                disps.append(center_now - center_prev)
        if disps:
            mean_disp = np.mean(disps, axis=0)
            velocity = mean_disp / total_dt

            # EMA smoothing
            if previous_velocity_map and label in previous_velocity_map:
                velocity = alpha * previous_velocity_map[label] + (1 - alpha) * velocity

            velocity_map[label] = velocity

    # Assign velocity to each point based on cluster label
    velocities = np.zeros_like(ref_frame)
    for idx, label in enumerate(cluster_labels):
        if label in velocity_map:
            velocities[idx] = velocity_map[label]
        else:
            velocities[idx] = 0

    speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
    angles = np.arctan2(velocities[:, 1], velocities[:, 0]).reshape(-1, 1)

    return np.hstack((ref_frame, speeds, angles)), velocity_map

def downsample_points(points, voxel_size=0.1):
    """
    Downsample the point cloud using a voxel grid filter.
    points: (N, 2) numpy array
    voxel_size: size of each voxel cell
    Returns:
        Downsampled point cloud as (M, 2) array
    """
    if points.shape[0] == 0:
        return points
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)
    voxel_dict = {}
    for idx, voxel in enumerate(voxel_indices):
        key = tuple(voxel)
        if key not in voxel_dict:
            voxel_dict[key] = []
        voxel_dict[key].append(points[idx])
    
    downsampled_points = np.array([np.mean(pts, axis=0) for pts in voxel_dict.values()])
    return downsampled_points


def get_velocity(lidar_scan, state, frame_buffer, time_buffer):
    if lidar_scan is None:
        return None, None
    transformed_points = transform_points(lidar_scan, state[:3])
    transformed_points = downsample_points(transformed_points)

    # Store frame and timestamp
    frame_buffer.append(transformed_points)
    time_buffer.append(time.time())

    previous_velocity_map = {}

    # Perform cluster-based multi-frame velocity estimation
    if len(frame_buffer) >= 2:
        velocities, previous_velocity_map = local_motion_estimation_cluster(
            list(frame_buffer), list(time_buffer), previous_velocity_map=previous_velocity_map
        )

        location = velocities[:, :2]
        velocity = np.column_stack((velocities[:, 2] * np.cos(velocities[:, 3]), velocities[:, 2] * np.sin(velocities[:, 3])))
        if location.shape[0] != 0:
            return location.T, velocity.T
    return None, None
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
import time

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

def local_motion_estimation_cluster(frames, times, eps=1, min_samples=2, previous_velocity_map=None, alpha=0.7):
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
    centroids_now = {
        label: ref_frame[cluster_labels == label].mean(axis=0)
        for label in set(cluster_labels) if label != -1
    }
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
            if previous_velocity_map and label in previous_velocity_map:
                velocity = alpha * previous_velocity_map[label] + (1 - alpha) * velocity
            velocity_map[label] = velocity

    velocities = np.zeros_like(ref_frame)
    for idx, label in enumerate(cluster_labels):
        if label in velocity_map:
            velocities[idx] = velocity_map[label]
        else:
            velocities[idx] = 0

    speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
    angles = np.arctan2(velocities[:, 1], velocities[:, 0]).reshape(-1, 1)
    return np.hstack((ref_frame, speeds, angles)), velocity_map

class KalmanFilter:
    def __init__(self, dt=1.0):
        self.dt = dt
        self.A = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0 ],
                           [0, 0, 0, 1 ]])
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        self.Q = np.eye(4) * 0.01
        self.R = np.eye(2) * 0.5
        self.P = np.eye(4)
        self.state = np.zeros((4, 1))

    def initialize(self, x, y, vx=0, vy=0):
        self.state = np.array([[x], [y], [vx], [vy]])

    def predict(self):
        self.state = self.A @ self.state
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, z):
        z = np.array([[z[0]], [z[1]]])
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

class Tracker:
    def __init__(self):
        self.kf_map = {}

    def update(self, positions, velocities, labels):
        current_labels = set()
        filtered_pos, filtered_vel = [], []

        for idx, label in enumerate(labels):
            if label == -1:
                continue
            current_labels.add(label)
            pos, vel = positions[idx], velocities[idx]

            if label not in self.kf_map:
                kf = KalmanFilter()
                kf.initialize(pos[0], pos[1], vel[0], vel[1])
                self.kf_map[label] = kf
            else:
                kf = self.kf_map[label]
                kf.predict()
                kf.update(pos)

            state = self.kf_map[label].state
            filtered_pos.append(state[:2].flatten())
            filtered_vel.append(state[2:].flatten())

        self.kf_map = {label: self.kf_map[label] for label in current_labels if label in self.kf_map}
        return np.array(filtered_pos), np.array(filtered_vel)

from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.means_ = np.array([[0.0], [0.1], [-0.1]])
gmm.covariances_ = np.array([[[0.002]], [[0.002]], [[0.002]]])
gmm.weights_ = np.array([0.4, 0.3, 0.3])
gmm.precisions_cholesky_ = np.array([[[1/np.sqrt(0.002)]]]*3)

def add_scattered_points(transformed_points, velocities, num_steps=5, step_size=0.1, num_samples=3):
    predicted_points = []
    predicted_velocities = []

    for i, (pos, vel) in enumerate(zip(transformed_points, velocities)):
        speed = np.linalg.norm(vel)
        if speed > 0.5:
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

frame_buffer = []
time_buffer = []
tracker = Tracker()
previous_velocity_map = {}

def process_lidar_frame(lidar_scan, state):
    global frame_buffer, time_buffer, tracker, previous_velocity_map
    if lidar_scan is None:
        return None, None
    transformed_points = transform_points(lidar_scan, state[:3])
    frame_buffer.append(transformed_points)
    time_buffer.append(time.time())

    if len(frame_buffer) > 10:
        frame_buffer.pop(0)
        time_buffer.pop(0)

    if len(frame_buffer) >= 2:
        velocities, previous_velocity_map = local_motion_estimation_cluster(
            frame_buffer, time_buffer, previous_velocity_map=previous_velocity_map)

        positions = velocities[:, :2]
        speeds = velocities[:, 2]
        angles = velocities[:, 3]
        vel_vectors = np.column_stack((speeds * np.cos(angles), speeds * np.sin(angles)))

        if positions.shape[0] == 0:
            return None, None

        labels = DBSCAN(eps=1, min_samples=2).fit_predict(positions)
        filtered_pos, filtered_vel = tracker.update(positions, vel_vectors, labels)

        transformed_points, velocities = add_scattered_points(filtered_pos, filtered_vel)

        return filtered_pos.T, filtered_vel.T
    return None, None
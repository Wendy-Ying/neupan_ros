import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree

def simulate_lidar_points():
    np.random.seed(0)
    wall1 = np.array([[2.0 + 0.01*np.random.randn(), y] for y in np.linspace(0, 2.0, 100)])
    wall2 = np.array([[3.0 + 0.01*np.random.randn(), y] for y in np.linspace(0, 2.0, 100)])
    
    people = []
    for px in [2.3, 2.5, 2.6, 2.65, 2.7]:
        for py in [0.5, 0.6, 0.8]:
            people.append([px + 0.01*np.random.randn(), py + 0.01*np.random.randn()])
    people = np.array(people)

    noise = np.random.uniform(low=[0, 0], high=[5, 5], size=(50, 2))

    return np.vstack((wall1, wall2, people, noise))
def find_elevator_goal(points, min_width=0.6, max_width=1.5):
    # cluster
    clustering = DBSCAN(eps=0.2, min_samples=5).fit(points)
    labels = clustering.labels_

    wall_clusters = []
    people_clusters = []

    for label in set(labels):
        if label == -1:
            continue
        cluster = points[labels == label]
        y_range = np.ptp(cluster[:, 1])
        x_range = np.ptp(cluster[:, 0])

        if y_range > 1.0 and x_range < 0.3:
            wall_clusters.append(cluster)
        elif 0.1 < x_range < 0.6 and 0.1 < y_range < 0.8 and len(cluster) < 50:
            people_clusters.append(cluster)

    if len(wall_clusters) < 2:
        return None, None, None, people_clusters

    for i in range(len(wall_clusters)):
        for j in range(i + 1, len(wall_clusters)):
            c1 = np.mean(wall_clusters[i], axis=0)
            c2 = np.mean(wall_clusters[j], axis=0)
            dx = abs(c1[0] - c2[0])
            if min_width <= dx <= max_width:
                left_x = min(c1[0], c2[0]) + 0.2
                right_x = max(c1[0], c2[0]) - 0.2
                y_range = (0.2, 1.5)

                x_candidates = np.linspace(left_x, right_x, 20)
                y_candidates = np.linspace(y_range[0], y_range[1], 10)
                grid = np.array([[x, y] for x in x_candidates for y in y_candidates])

                all_people_points = np.vstack(people_clusters) if people_clusters else np.empty((0, 2))
                if len(all_people_points) > 0:
                    people_tree = KDTree(all_people_points)
                    people_dists, _ = people_tree.query(grid)
                else:
                    people_dists = np.full(len(grid), np.inf)

                wall_points = np.vstack((wall_clusters[i], wall_clusters[j]))
                wall_tree = KDTree(wall_points)
                wall_dists, _ = wall_tree.query(grid)

                # Only accept points far from both people and walls
                valid_mask = (people_dists > 0.3) & (wall_dists > 0.25)
                free_points = grid[valid_mask]

                if len(free_points) > 0:
                    goal = free_points[np.argmin(np.abs(free_points[:, 1] - 1.0))]
                    return wall_clusters[i], wall_clusters[j], goal, people_clusters

    return None, None, None, people_clusters

def find_elevator_goal(points, min_width=0.6, max_width=1.5):
    clustering = DBSCAN(eps=0.2, min_samples=5).fit(points)
    labels = clustering.labels_

    wall_clusters = []
    people_clusters = []

    for label in set(labels):
        if label == -1:
            continue
        cluster = points[labels == label]
        y_range = np.ptp(cluster[:, 1])
        x_range = np.ptp(cluster[:, 0])

        if y_range > 1.0 and x_range < 0.3:
            wall_clusters.append(cluster)
        elif 0.1 < x_range < 0.6 and 0.1 < y_range < 0.8 and len(cluster) < 50:
            people_clusters.append(cluster)

    if len(wall_clusters) < 2:
        return None, None, None, people_clusters

    for i in range(len(wall_clusters)):
        for j in range(i + 1, len(wall_clusters)):
            c1 = np.mean(wall_clusters[i], axis=0)
            c2 = np.mean(wall_clusters[j], axis=0)
            dx = abs(c1[0] - c2[0])
            if min_width <= dx <= max_width:
                left_x = min(c1[0], c2[0]) + 0.2
                right_x = max(c1[0], c2[0]) - 0.2
                middle_x = (c1[0] + c2[0]) / 2.0
                y_range = (0.2, 1.5)

                x_candidates = np.linspace(left_x, right_x, 20)
                y_candidates = np.linspace(y_range[0], y_range[1], 10)
                grid = np.array([[x, y] for x in x_candidates for y in y_candidates])

                all_people_points = np.vstack(people_clusters) if people_clusters else np.empty((0, 2))
                people_dists = np.full(len(grid), np.inf)
                if len(all_people_points) > 0:
                    people_tree = KDTree(all_people_points)
                    people_dists, _ = people_tree.query(grid)

                wall_points = np.vstack((wall_clusters[i], wall_clusters[j]))
                wall_tree = KDTree(wall_points)
                wall_dists, _ = wall_tree.query(grid)

                center_deviation = (grid[:, 0] - middle_x)**2

                people_penalty = np.where(people_dists < 0.3, 1e3 * (0.3 - people_dists)**2, 0)
                wall_penalty = np.where(wall_dists < 0.25, 1e3 * (0.25 - wall_dists)**2, 0)

                scores = 1.0 * center_deviation + 1.0 * people_penalty + 1.0 * wall_penalty

                valid_mask = (people_dists > 0.2) & (wall_dists > 0.15)
                if np.any(valid_mask):
                    best_idx = np.argmin(scores + (~valid_mask) * 1e6)
                    goal = grid[best_idx]
                    return wall_clusters[i], wall_clusters[j], goal, people_clusters

    return None, None, None, people_clusters


def visualize(points, walls, people_clusters, goal):
    plt.figure(figsize=(8, 6))
    plt.scatter(points[:, 0], points[:, 1], c='lightgray', s=10, label='Lidar Points')
    for i, wall in enumerate(walls):
        plt.scatter(wall[:, 0], wall[:, 1], s=30, label=f'Wall {i+1}')
    for person in people_clusters:
        plt.scatter(person[:, 0], person[:, 1], c='orange', s=50, marker='o', label='Person')
    if goal is not None:
        plt.scatter(goal[0], goal[1], c='red', s=100, marker='x', label='Goal Point')
    plt.title("Elevator Detection")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    points = simulate_lidar_points()
    wall1, wall2, goal, people = find_elevator_goal(points)
    if goal is not None:
        print(f"Goal found at: {goal}")
        visualize(points, [wall1, wall2], people, goal)
    else:
        print("No suitable goal found.")
        visualize(points, [], people, None)

if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA

def change_numpy(file_path):
    """454/-420/3 change to numpy array"""
    data = np.loadtxt(file_path, delimiter='/')
    return data

def sc_pca(data):
    pca = PCA(n_components=2)
    pca.fit(data)
    transformed_data = pca.transform(data)
    return transformed_data

def compute_features(points):
    # 1) closedness
    diffs = points[1:] - points[:-1]
    seg_len = np.linalg.norm(diffs, axis=1)
    path_len = float(seg_len.sum())
    start_end = float(np.linalg.norm(points[0] - points[-1]))
    closed = start_end / (path_len + 1e-8)

    # 2) line_ratio (공분산 고유값 비율)
    points_c = points - points.mean(axis=0)
    cov = np.cov(points_c.T)
    eigvals, _ = np.linalg.eigh(cov)
    eigvals = np.sort(eigvals)
    lam_small, lam_large = eigvals[0], eigvals[1]
    line_r = float(lam_small / (lam_large + 1e-8))

    # 3) angle_cov_ratio (중심 = 평균 기준)
    center = points.mean(axis=0)
    rel = points - center
    theta = np.arctan2(rel[:, 1], rel[:, 0])
    theta = np.mod(theta, 2 * np.pi)

    theta_sorted = np.sort(theta)
    dtheta = np.diff(theta_sorted)
    wrap_gap = 2 * np.pi - (theta_sorted[-1] - theta_sorted[0])
    dtheta = np.concatenate([dtheta, [wrap_gap]])

    max_gap = float(dtheta.max())
    angle_cov = 2 * np.pi - max_gap
    angle_ratio = float(angle_cov / (2 * np.pi + 1e-8))

    return closed, line_r, angle_ratio

def circle_classfication(points,
                closed_thresh=0.25,
                angle_thresh=0.8,
                line_ratio_thresh=0.10):
    closed, line_r, angle_ratio = compute_features(points)
    # 1) 충분히 닫혀 있고
    cond_closed = closed <= closed_thresh

    # 2) 거의 한 바퀴 돌았고
    cond_angle = angle_ratio >= angle_thresh

    # 3) 완전 직선은 아니어야 함
    cond_not_line = line_r  >= line_ratio_thresh

    is_circle = bool(cond_closed and cond_angle and cond_not_line)
    return is_circle

# def visualize_points_with_center(points):
#     center = points.mean(axis=0)
#     plt.figure(figsize=(6,6))
#     plt.scatter(points[:,0], points[:,1], label='Points', alpha=0.7)
#     plt.scatter(center[0], center[1], color='red', label='Center', s=100, marker='X')
#     plt.axis('equal')
#     plt.legend()
#     plt.title('Points and Center')
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.show()

# file_path = "/Users/parksung-cheol/Desktop/ml_project-doki/augmented_data/circle/2__aug003.txt"
# data = change_numpy(file_path)
# transformed_data = sc_pca(data)
# visualize_points_with_center(transformed_data)

# file_path = "/Users/parksung-cheol/Desktop/ml_project-doki/augmented_data/vertical/2__aug003.txt"
# data = change_numpy(file_path)
# transformed_data = sc_pca(data)
# closed, line_r, angle_ratio = compute_features(transformed_data)
# print("Closedness:", closed)
# print("Line Ratio:", line_r)
# print("Angle Covariance Ratio:", angle_ratio)
# print(circle_classfication(transformed_data))

# folder_path = "/Users/parksung-cheol/Desktop/ml_project-doki/augmented_data/horizontal"

# stack = []
# for filename in os.listdir(folder_path):
#     if filename.endswith(".txt"):
#         file_path = os.path.join(folder_path, filename)
#         data = change_numpy(file_path)
#         transformed_data = sc_pca(data)
#         result = circle_classfication(transformed_data)
#         print(result)
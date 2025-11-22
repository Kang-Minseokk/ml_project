import numpy as np
import matplotlib.pyplot as plt
import os

def change_numpy_yz(file_path):
    """454/-420/3 change to numpy array"""
    data = np.loadtxt(file_path, delimiter='/')
    yz_data = data[:, 1:3]
    return yz_data

def fit_line(points):
    """
    points: (N, 2) 형태의 numpy 배열 또는 리스트 [[x1,y1], [x2,y2], ...]
    반환: a, b  (y = a x + b)
    """
    x = points[:, 0]
    y = points[:, 1]
    dx = x.max() - x.min()
    dy = y.max() - y.min()

    a = dy / (dx + 1e-8)  # 기울기

    return a

def plus_determine(points):
    x = points[:, 0]
    y = points[:, 1]

    a, b = np.polyfit(x, y, 1)
    return a, b

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

# file_path = "/Users/parksung-cheol/Desktop/ml_project-doki/augmented_data/vertical/4__aug003.txt"
# data = change_numpy_yz(file_path)
# visualize_points_with_center(data)

# file_path = "/Users/parksung-cheol/Desktop/ml_project-doki/augmented_data/vertical/4__aug003.txt"
# data = change_numpy_yz(file_path)
# a = fit_line(data)
# print(a)

# folder_path = "/Users/parksung-cheol/Desktop/ml_project-doki/augmented_data/diagonal_right"

# stack = []
# for filename in os.listdir(folder_path):
#     if filename.endswith(".txt"):
#         file_path = os.path.join(folder_path, filename)
#         data = change_numpy_yz(file_path)
#         a = fit_line(data)
#         stack.append(a)
# print("diagonal_right")
# print(max(stack))
# print(min(stack))

# folder_path = "/Users/parksung-cheol/Desktop/ml_project-doki/augmented_data/diagonal_right"

# stack = []
# for filename in os.listdir(folder_path):
#     if filename.endswith(".txt"):
#         file_path = os.path.join(folder_path, filename)
#         data = change_numpy_yz(file_path)
#         a, b = plus_determine(data)
#         stack.append(a)
# print("diagonal_right")
# print(max(stack))
# print(min(stack))
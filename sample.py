import glob
import numpy as np

def load_trajectory_file(path):
    xs, ys, zs = [], [], []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) <= 6:
                continue
            xyz_str = parts[6]
            xyz = xyz_str.split('/')
            if len(xyz) != 3:
                continue
            try:
                x, y, z = map(float, xyz)
                xs.append(x)
                ys.append(y)
                zs.append(z)
            except ValueError:
                continue
    return np.array(xs), np.array(ys), np.array(zs)  # shape (T,), (T,), (T,)

# 예시: 1.txt만 먼저 읽어보기
xs, ys, zs = load_trajectory_file("1st_data/circle/1.txt")
print(xs.shape, ys.shape, zs.shape)

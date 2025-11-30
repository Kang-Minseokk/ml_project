import numpy as np
import os
from sklearn.decomposition import PCA

# ---- 초기 데이터 파일 병합 -----
def collect_txt_files(root_dir):
    txt_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for name in filenames:
            if name.lower().endswith(".txt"):
                full_path = os.path.join(dirpath, name)
                txt_files.append(full_path)
    txt_files.sort()
    return txt_files

def merge_all_txt(root_dir, out_file):
    txt_files = collect_txt_files(root_dir)

    with open(out_file, "w", encoding="utf-8") as out_f:
        for path in txt_files:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    out_f.write(line.rstrip("\n") + "\n")
# -------------------------------------

def data_transform(file_path, prefix='data'):
    """데이터 전처리"""
    blocks = []
    current_block = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith('s'):
                if current_block:
                    blocks.append(current_block)
                    current_block = []
                continue

            cols = line.split(',')
            if len(cols) > 6 and cols[6] != '':
                current_block.append(cols[6])

    if current_block:
        blocks.append(current_block)

    for i, block in enumerate(blocks, start=1):
        out_path = f'data/{prefix}{i}.txt'
        with open(out_path, 'w') as out_f:
            out_f.write('\n'.join(block))

    return blocks

# ---- 성철 코드 -----
class Sungcheol:
    # ---- 원과 직선을 결정하는 코드-----
    @staticmethod
    def change_numpy(file_path):
        """기훈님 파일에서 행렬 형식으로 변환"""
        data = np.loadtxt(file_path, delimiter='/')
        return data
    
    @staticmethod
    def sc_pca(data):
        """ 주성분 분석 (PCA) 수행"""
        pca = PCA(n_components=2)
        pca.fit(data)
        transformed_data = pca.transform(data)
        return transformed_data

    @staticmethod
    def compute_features(points):
        # 1) closedness : 닫힘정도
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

    @staticmethod
    def circle_classfication(points,
                    closed_thresh=0.25,
                    angle_thresh=0.8,
                    line_ratio_thresh=0.10):
        closed, line_r, angle_ratio = Sungcheol.compute_features(points)
        # 1) 충분히 닫혀 있고
        cond_closed = closed <= closed_thresh

        # 2) 거의 한 바퀴 돌았고
        cond_angle = angle_ratio >= angle_thresh

        # 3) 완전 직선은 아니어야 함
        cond_not_line = line_r  >= line_ratio_thresh

        is_circle = bool(cond_closed and cond_angle and cond_not_line)
        return is_circle
    # -------------------------------------

    # ---- 직선 기울기 구하는 코드 -----
    @staticmethod
    def change_numpy_yz(file_path):
        """454/-420/3 change to numpy array"""
        data = np.loadtxt(file_path, delimiter='/')
        yz_data = data[:, 1:3]
        return yz_data

    @staticmethod
    def fit_line(points):
        """ 직선의 절댓값 기울기 반환 """
        x = points[:, 0]
        y = points[:, 1]
        dx = x.max() - x.min()
        dy = y.max() - y.min()

        a = dy / (dx + 1e-8)

        return a
    
    @staticmethod
    def plus_determine(points):
        """회귀분석 직선의 기울기와 절편을 반환"""
        x = points[:, 0]
        y = points[:, 1]

        a, b = np.polyfit(x, y, 1)
        return a, b

# -------------------------------------
    @staticmethod
    def circle_check(file_path):
        data = Sungcheol.change_numpy(file_path)
        transformed_data = Sungcheol.sc_pca(data)
        is_circle = Sungcheol.circle_classfication(transformed_data)
        if is_circle:
            return 'circle'

    @staticmethod
    def line_check(file_path):
        data = Sungcheol.change_numpy_yz(file_path)
        a = Sungcheol.fit_line(data)
        if abs(a) < 0.3:
            return 'horizontal'
        elif abs(a) > 4:
            return 'vertical'
        else:
            a, b = Sungcheol.plus_determine(data)
            if a < 0:
                return 'diagonal_right'
            else:
                return 'diagonal_left'

# -------------------------------------

# ---- 민석 코드 ----
class Minseok:
    @staticmethod
    def make_list(file_path):
        x_list, y_list, z_list = [], [], []
        with open(file_path, "r") as f :
            while True:
                line = f.readline()
                if not line :
                    break
                x, y, z = line.split('/')
                x_list.append(int(x))
                y_list.append(int(y))
                z_list.append(int(z))

        return x_list, y_list, z_list
    
    @staticmethod
    def check_range(x_list, y_list, z_list):
        max_x, max_y, max_z = max(x_list), max(y_list), max(z_list)
        min_x, min_y, min_z = min(x_list), min(y_list), min(z_list)
        x_range, y_range, z_range = max_x - min_x, max_y - min_y, max_z - min_z
        return x_range, y_range, z_range
    
    @staticmethod
    def pca_analysis(x_list, y_list, z_list):
        """
        PCA 기반으로 궤적 형태를 판별합니다.
        
        return:
        - "line"   : 거의 직선 궤적
        - "circle" : 평면상 곡선(원 가능성 높음)
        """
        points = np.column_stack([x_list, y_list, z_list])
        mean = points.mean(axis=0)
        centered = points - mean

        U, S, Vt = np.linalg.svd(centered)

        # 방향성 분산 체크
        ratio = S[1] / S[0]        
        
        return ratio  
    
    @staticmethod
    def minseok_circle(file_path):
        x_list, y_list, z_list = Minseok.make_list(file_path)
        ratio = Minseok.pca_analysis(x_list=x_list, y_list=y_list, z_list=z_list)

        RATIO_THRESHOLD=0.5    
        if ratio > RATIO_THRESHOLD :
            return "circle"
        else :
            return  "linear" 
    
    @staticmethod
    def minseok_check1(file_path):
        x_list, y_list, z_list = Minseok.make_list(file_path)
        x_range, y_range, z_range = Minseok.check_range(x_list, y_list, z_list)

        if (z_range < x_range) and (z_range < y_range) :        
            return "horizontal"
        elif (y_range < x_range) and (y_range < z_range) :
            return "vertical"
        else :
            return "unknown"
        
    @staticmethod
    def minseok_check2(file_path):
        x_list, y_list, z_list = Minseok.make_list(file_path)

        middle_y_idx = len(y_list) // 2
        middle_z_idx = len(z_list) // 2

        middle_y_val = y_list[middle_y_idx]
        middle_z_val = z_list[middle_z_idx]

        slope_y = middle_y_val - y_list[0]
        slope_z = middle_z_val - z_list[0]

        slope = slope_z / slope_y       
        if slope > 0 :
            return "diagonal_left"
        else:   
            return "diagonal_right"

# -------------------------------------

# ---- 재은 코드 ----
class Jaeun:
    pass
# -------------------------------------

file_path = "raw_data"
out_path = "final_data.txt"

# merge_all_txt(file_path, out_path) # 데이터 병합

# data_transform(out_path, prefix='data') # 데이터 전처리

def main():
    path = "data"
    file_list = sorted(os.listdir(path))

    for filename in file_list:
        if filename.endswith(".txt"):
            file_path = os.path.join(path, filename)
            sungcheol_shape = Sungcheol.circle_check(file_path)
            if sungcheol_shape is None:
                sungcheol_shape = Sungcheol.line_check(file_path)

            # print("성철이 결과")
            # print(f"{filename}: {sungcheol_shape}")

            minseok_shape = Minseok.minseok_circle(file_path)
            if minseok_shape == "linear":
                minseok_shape = Minseok.minseok_check1(file_path)
                if minseok_shape == "unknown":
                    minseok_shape = Minseok.minseok_check2(file_path)

            # print("민석이 형 결과")
            # print(f"{filename}: {minseok_shape}")

if __name__ == "__main__": main()

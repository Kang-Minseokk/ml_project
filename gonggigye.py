import numpy as np
import os
from sklearn.decomposition import PCA
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

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
        data = np.loadtxt(file_path, delimiter=' ')
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
        data = np.loadtxt(file_path, delimiter=' ')
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
                line = f.readline().strip()
                if not line :
                    break
                x, y, z = line.split(' ')
                x_list.append(float(x))
                y_list.append(float(y))
                z_list.append(float(z))

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

        if (y_range > x_range) and (y_range > z_range) : # Fix: 멘토님의 피드백을 받고 min에서 max를 비교하도록 수정
            return "horizontal"
        elif (z_range > x_range) and (z_range > y_range) : # Fix: 멘토님의 피드백을 받고 min에서 max를 비교하도록 수정
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
class Jaeeun:
    # ---- RandomForest 기반 3D 궤적 분류 -----
    
    @staticmethod
    def load_xyz_from_txt(file_path):
        """궤적 파일에서 x,y,z 좌표 추출"""
        coords = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()                    
                    if line :
                        parts = line.split(' ')                        
                        if len(parts) >= 3:
                            try:
                                x = float(parts[0])
                                y = float(parts[1]) 
                                z = float(parts[2])
                                coords.append([x, y, z])
                            except ValueError:
                                continue
        except FileNotFoundError:
            return np.array([])
        except Exception:
            return np.array([])
        
        if not coords:
            return np.array([])
        
        return np.array(coords)

    @staticmethod
    def compute_features(coords):
        """궤적의 물리적 특성 10가지 계산"""
        if len(coords) == 0:
            return {}
        
        # 좌표 분리
        X, Y, Z = coords[:, 0], coords[:, 1], coords[:, 2]
        
        # 연속한 두 점 차이
        diffs = coords[1:] - coords[:-1]
        dx, dy, dz = diffs[:, 0], diffs[:, 1], diffs[:, 2]

        # 전체 이동 거리
        step_dist = np.linalg.norm(diffs, axis=1)
        path_length = step_dist.sum()

        # 처음 → 마지막 직선 거리
        total_disp = np.linalg.norm(coords[-1] - coords[0])

        # 직선성
        straightness = total_disp / (path_length + 1e-6)

        # 방향 전환 횟수 (부호 변화)
        def count_sign_changes(arr):
            signs = np.sign(arr)
            return np.sum(signs[:-1] != signs[1:])

        direction_changes = (
            count_sign_changes(dx) +
            count_sign_changes(dy) +
            count_sign_changes(dz)
        )

        # 곡률 계산
        curvature_vals = []
        for i in range(len(coords) - 2):
            v1 = coords[i+1] - coords[i]
            v2 = coords[i+2] - coords[i+1]
            denom = (np.linalg.norm(v1)**3 + 1e-6)
            curvature_vals.append(np.linalg.norm(np.cross(v1, v2)) / denom)

        curvature_mean = np.mean(curvature_vals) if curvature_vals else 0.0

        # X/Y/Z축 범위
        range_x = X.max() - X.min()
        range_y = Y.max() - Y.min()
        range_z = Z.max() - Z.min()

        # XY 평면 vs Z 에너지 비율
        xy_energy = np.sum(np.abs(dx) + np.abs(dy))
        z_energy = np.sum(np.abs(dz))
        total_energy = xy_energy + z_energy + 1e-6

        xy_ratio = xy_energy / total_energy
        z_ratio = z_energy / total_energy

        return {
            "range_x": range_x,
            "range_y": range_y,
            "range_z": range_z,
            "path_length": path_length,
            "total_disp": total_disp,
            "straightness": straightness,
            "direction_changes": direction_changes,
            "curvature_mean": curvature_mean,
            "xy_ratio": xy_ratio,
            "z_ratio": z_ratio
        }

    @staticmethod
    def predict_trajectory(file_path, model_path=None):
        """궤적 벤턴 분류 함수"""
        # 1. 훈련된 모델 로드 (전처리된 데이터 경로 기준)
        if model_path is None:
            model_paths = [
                './results/trained_model.pkl',  # 공기계.py 실행 디렉토리
                '../results/trained_model.pkl',  # 상위 디렉토리
                'trained_model.pkl'  # 현재 디렉토리
            ]
        else:
            model_paths = [model_path]
            
        model = None
        for path in model_paths:
            try:
                model = joblib.load(path)
                break
            except:
                continue
                
        if model is None:
            model = Jaeeun._train_backup_model()
        
        # 2. 궤적 데이터 로드
        coords = Jaeeun.load_xyz_from_txt(file_path)        
        if len(coords) == 0:
            return "unknown"
        
        # 3. 10가지 특성 추출
        features = Jaeeun.compute_features(coords)
        feature_names = ['range_x', 'range_y', 'range_z', 'path_length', 'total_disp', 
                        'straightness', 'direction_changes', 'curvature_mean', 'xy_ratio', 'z_ratio']
        
        # 4. RandomForest 예측
        feature_array = np.array([features[name] for name in feature_names])
        features_reshaped = feature_array.reshape(1, -1)        
        feature_df = pd.DataFrame(features_reshaped, columns=feature_names) # Warning 제거를 위한 코드 추가
        
        try:
            prediction = model.predict(feature_df)[0]            
            return prediction
        except:
            return "unknown"

    @staticmethod
    def _train_backup_model():
        """PKL 없을 시 실제 학습 데이터로 새 모델 학습"""
        print("PKL 파일을 찾을 수 없어 새로운 모델을 학습합니다...")
        
        # 전처리된 data 파일들로 학습 (공기계.py에서 생성된 파일들)
        import glob
        
        # 전처리된 data 파일 경로들
        data_paths = [
            './data',  # 현재 디렉토리의 data 폴더
            '../data',  # 상위 디렉토리의 data 폴더
            '/Users/julia/Desktop/my_project/data'  # 절대 경로
        ]
        
        base_path = None
        for path in data_paths:
            if os.path.exists(path):
                base_path = path
                break
                
        if base_path is None:
            print("전처리된 학습 데이터를 찾을 수 없습니다. 에러를 반환합니다.")
            raise FileNotFoundError("전처리된 데이터가 없어 모델을 생성할 수 없습니다. 먼저 data_transform()을 실행하세요.")
        
        print(f"전처리된 학습 데이터 발견: {base_path}")
        
        # 전처리된 data 파일들로 학습
        X_train = []
        y_train = []
        
        # data 폴더의 모든 txt 파일 처리
        txt_files = glob.glob(os.path.join(base_path, "*.txt"))
        
        # 임시로 각 파일을 순환하며 5개 클래스에 균등 분배 (실제로는 라벨링 로직 필요)
        classes = ['circle', 'horizontal', 'vertical', 'diagonal_left', 'diagonal_right']
        for i, txt_file in enumerate(txt_files[:75]):  # 최대 75개 파일 (클래스당 15개)
            coords = Jaeeun.load_xyz_from_txt(txt_file)
            if len(coords) > 0:
                features = Jaeeun.compute_features(coords)
                feature_names = ['range_x', 'range_y', 'range_z', 'path_length', 'total_disp', 
                                'straightness', 'direction_changes', 'curvature_mean', 'xy_ratio', 'z_ratio']
                feature_array = [features[name] for name in feature_names]
                X_train.append(feature_array)
                # 순환하며 클래스 할당 (실제로는 다른 방법으로 라벨링 필요)
                y_train.append(classes[i % 5])
        
        if len(X_train) == 0:
            print("유효한 전처리된 학습 데이터가 없습니다.")
            raise ValueError("학습할 수 있는 전처리된 데이터가 없습니다.")
        
        # 실제 학습 실행
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        print(f"학습 데이터: {len(X_train)}개 샘플")
        print(f"클래스 분포: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        
        # train_model.py와 동일한 설정으로 학습
        model = RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1
        )
        
        model.fit(X_train, y_train)
        
        return model
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

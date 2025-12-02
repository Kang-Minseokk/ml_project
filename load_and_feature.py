import numpy as np
import pandas as pd
import glob
import os


# ---------------------------------------------------------
# TXT 파일 하나에서 로봇 팔의 (X, Y, Z) 좌표들을 읽어오는 함수
#    → 각 줄은 "x/y/z" 형태로 되어 있음
#    → 한 파일 = 한 동작(trajectory)
#    → 출력: (N, 3) 모양의 numpy 배열  ← 좌표들
# ---------------------------------------------------------
def load_xyz_from_txt(path):
    coords = []  # 좌표들을 저장할 리스트
    
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()     # 앞뒤 공백 제거 
            if not line:
                continue            # 빈 줄이면 skip
            
            if line.startswith('s'):
                continue            # s로 시작하는 line은 제외시키기
                        
            line = line.split(',')[6]                     

            try:
                # "x/y/z" 구조에서 '/' 기준으로 나눠서 float로 변환                
                x, y, z = map(float, line.split('/'))
                coords.append([x, y, z])                               
            except ValueError:
                # 숫자가 아닌 줄이 들어가 있을 가능성을 대비
                print(f"Warning: Invalid line in file {path}: {line}")
                continue

    return np.array(coords)  # N개의 좌표가 담긴 (N,3) 배열 반환



# ---------------------------------------------------------
# 좌표(trajectory) 하나를 입력받아 “설명 가능한 feature”들을 계산하는 함수
#    feature = 모델의 입력으로 사용되는 숫자 특성들
#
#    이 함수가 중요함 → raw XYZ 좌표를 사람이 이해할 수 있는 값으로 변환
# ---------------------------------------------------------
def compute_features(coords):
    if coords.size == 0:
        raise ValueError("Input coordinates are empty. Check the data files.")

    # 좌표를 X, Y, Z 각각 따로 분리
    X = coords[:, 0]
    Y = coords[:, 1]
    Z = coords[:, 2]

    # -----------------------------------------------------
    # 1) 연속한 두 점 차이(diff)
    #    → 한 스텝마다 얼마나 움직였는지 계산
    # -----------------------------------------------------
    diffs = coords[1:] - coords[:-1]
    dx, dy, dz = diffs[:, 0], diffs[:, 1], diffs[:, 2]

    # -----------------------------------------------------
    # 2) 전체 이동 거리(path_length)
    #    step_dist = 한 스텝에서의 이동량
    # -----------------------------------------------------
    step_dist = np.linalg.norm(diffs, axis=1)
    path_length = step_dist.sum()           # 전체 이동 거리의 합

    # -----------------------------------------------------
    # 3) 처음 위치 → 마지막 위치까지의 직선 거리(displacement)
    # -----------------------------------------------------
    total_disp = np.linalg.norm(coords[-1] - coords[0])

    # -----------------------------------------------------
    # 4) 직선성(straightness)
    #    straightness = (직선 거리) / (전체 이동거리)
    #    → 1에 가까우면 거의 직선
    #    → circle처럼 빙빙 돌면 작아짐
    # -----------------------------------------------------
    straightness = total_disp / (path_length + 1e-6)

    # -----------------------------------------------------
    # 5) 방향 전환 횟수(direction_changes)
    #    dx, dy, dz 각각에서 부호가 바뀌는 횟수 확인
    #    → 움직이는 방향이 얼마나 자주 바뀌었는지
    # -----------------------------------------------------
    def count_sign_changes(arr):
        signs = np.sign(arr)
        return np.sum(signs[:-1] != signs[1:])

    direction_changes = (
        count_sign_changes(dx) +
        count_sign_changes(dy) +
        count_sign_changes(dz)
    )

    # -----------------------------------------------------
    # 6) 곡률(curvature) 계산
    #    연속한 세 점을 이용하여 얼마나 “꺾였는지” 측정
    #    circle일수록 곡률이 큼, 직선은 거의 0
    # -----------------------------------------------------
    curvature_vals = []
    for i in range(len(coords) - 2):
        v1 = coords[i+1] - coords[i]
        v2 = coords[i+2] - coords[i+1]
        denom = (np.linalg.norm(v1)**3 + 1e-6)
        curvature_vals.append(np.linalg.norm(np.cross(v1, v2)) / denom)

    curvature_mean = np.mean(curvature_vals) if curvature_vals else 0.0

    # -----------------------------------------------------
    # 7) X/Y/Z축 범위(range_x, range_y, range_z)
    #    → 그 방향으로 얼마나 퍼져있는지(덜컥한 정도)
    # -----------------------------------------------------
    range_x = X.max() - X.min()
    range_y = Y.max() - Y.min()
    range_z = Z.max() - Z.min()

    # -----------------------------------------------------
    # 8) XY 평면 이동량 vs Z 이동량 비율(xy_ratio, z_ratio)
    #    → 평면 위에서 많이 놀았는지? 위아래(Z)로 많이 움직였는지?
    # -----------------------------------------------------
    xy_energy = np.sum(np.abs(dx) + np.abs(dy))
    z_energy = np.sum(np.abs(dz))
    total_energy = xy_energy + z_energy + 1e-6

    xy_ratio = xy_energy / total_energy
    z_ratio  = z_energy / total_energy

    # -----------------------------------------------------
    # 9) YZ 평면 기울기 (diagonal 구분 개선용)
    #    → Y에 대한 Z의 선형 회귀 기울기
    #    → diagonal_left (양의 기울기) vs diagonal_right (음의 기울기)
    # -----------------------------------------------------
    yz_slope = calculate_yz_slope(Y, Z)
    
    # -----------------------------------------------------
    # 10) YZ 상관계수 (diagonal 구분 보조용)
    #     → Y와 Z가 얼마나 선형적으로 연관되는지
    # -----------------------------------------------------
    yz_correlation = calculate_yz_correlation(Y, Z)

    # 하나의 trajectory에 대한 feature dict 반환 (12개 특성)
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
        "z_ratio": z_ratio,
        "yz_slope": yz_slope,
        "yz_correlation": yz_correlation
    }


def calculate_yz_slope(Y, Z):
    """YZ 평면에서의 회귀 직선 기울기 계산 (diagonal 구분용)"""
    if len(Y) < 2 or np.std(Y) < 1e-6:
        return 0.0
    
    # Y에 대한 Z의 선형 회귀 기울기
    slope, _ = np.polyfit(Y, Z, 1)
    return slope

def calculate_yz_correlation(Y, Z):
    """YZ 좌표간 피어슨 상관계수 계산 (diagonal 구분용)"""
    if len(Y) < 2:
        return 0.0
    
    # 표준편차가 0에 가까우면 상관계수 계산 불가
    if np.std(Y) < 1e-6 or np.std(Z) < 1e-6:
        return 0.0
    
    correlation = np.corrcoef(Y, Z)[0, 1]
    return correlation if not np.isnan(correlation) else 0.0


# ---------------------------------------------------------
# 여러 폴더에 있는 모든 txt 파일들을 읽어서
#    → compute_features로 feature 계산하고
#    → 하나의 큰 DataFrame으로 묶는 함수
#
#    최종 출력: df (행 = 파일 하나, 열 = feature + label + file이름)
# ---------------------------------------------------------
def build_dataset(root_folder):
    rows = []

    # 각 클래스명 = 폴더 이름
    class_names = ['horizontal', 'vertical', 'diagonal_left', 'diagonal_right', 'circle']

    for label in class_names:
        folder = os.path.join(root_folder, label)        # e.g., "./augmented_data/horizontal"
        files = glob.glob(f"{folder}/*.txt")             # 해당 폴더 안의 모든 .txt 파일

        for path in files:
            coords = load_xyz_from_txt(path)             # txt → XYZ 좌표

            if coords.size == 0:
                print(f"Warning: File {path} contains no valid data and will be skipped.")
                continue

            feats = compute_features(coords)             # 좌표 → feature 계산
            feats['label'] = label                      # 정답 라벨
            feats['file'] = os.path.basename(path)       # 파일 이름 기록
            rows.append(feats)                           # 한 줄(row)로 기록

    return pd.DataFrame(rows)                            # 전체 데이터프레임 반환
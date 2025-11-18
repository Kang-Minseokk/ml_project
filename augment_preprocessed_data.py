# augment_preprocessed.py
from __future__ import annotations
from pathlib import Path
import os
import math
import random
import numpy as np
from typing import List, Tuple

# -----------------------------
# 설정 (필요시 숫자만 바꾸면 됨)
# -----------------------------
RNG_SEED = 42
SRC_DIR = "preprocessed_data"
OUT_DIR = "augmented_data"

# 카테고리별 "이번 주 받은 (노이즈 포함) 두 번째 데이터" 파일 구간
NOISY_RANGES = {
    "circle":         [(9, 16)],
    "diagonal_left":  [(8, 12)],
    "diagonal_right": [(8, 12)],
    "horizontal":     [(7, 11)],
    "vertical":       [(7, 11)],
}

# 증강 개수: (클린, 노이즈) 각각 한 원본당 생성할 변형 수
N_AUG_PER_CLEAN = 3
N_AUG_PER_NOISY = 5

# 증강 강도 파라미터
JITTER_STD_FRAC_CLEAN = 0.01   # 좌표 표준편차 대비 가우시안 노이즈 비율(클린)
JITTER_STD_FRAC_NOISY = 0.005  # 스무딩 이후 미세 노이즈(노이즈 샘플)

ROT_DEG_MAX_CLEAN = 12         # 무작위 회전 각도(도)
ROT_DEG_MAX_NOISY = 8

SCALE_RANGE = (0.95, 1.05)     # 등방성 스케일
TRANSLATE_STD_FRAC = 0.02      # 좌표 표준편차 대비 이동량 표준편차

SMOOTH_WINDOW = 7              # 이동평균 윈도(홀수 권장)
MIN_POINTS = 8                 # 너무 짧은 궤적은 스킵

# -----------------------------
# 유틸
# -----------------------------
def is_noisy_file(category: str, stem_num: int) -> bool:
    ranges = NOISY_RANGES.get(category, [])
    return any(lo <= stem_num <= hi for (lo, hi) in ranges)

def read_xyz_txt(path: Path) -> np.ndarray:
    xs, ys, zs = [], [], []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                x, y, z = line.split("/")
                xs.append(float(x)); ys.append(float(y)); zs.append(float(z))
            except Exception:
                continue
    if len(xs) == 0:
        return np.empty((0, 3))
    return np.stack([np.array(xs), np.array(ys), np.array(zs)], axis=1)  # (T,3)

def write_xyz_txt(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for x, y, z in arr:
            f.write(f"{int(round(x))}/{int(round(y))}/{int(round(z))}\n")

def moving_average(a: np.ndarray, window: int) -> np.ndarray:
    """(T,3) -> (T,3) 간단한 이동평균 스무딩. 윈도는 홀수 추천."""
    if window <= 1 or a.shape[0] < window:
        return a.copy()
    pad = window // 2
    # 가장자리 보존을 위해 양끝을 가장자리 값으로 패딩
    padded = np.pad(a, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones((window,)) / window
    out = np.zeros_like(a)
    for d in range(3):
        out[:, d] = np.convolve(padded[:, d], kernel, mode="valid")
    return out

def standardize_trajectory(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """평균/표준편차 반환 (이동/스케일 증강용 기준치 계산)"""
    mean = a.mean(axis=0)
    std = a.std(axis=0) + 1e-9
    return mean, std, a

def rand_rotation_matrix(max_deg: float) -> np.ndarray:
    """무작위 소각 회전행렬(라디안으로 변환 후 Z-Y-X 순 회전)"""
    def deg2rad(d): return d * math.pi / 180.0
    ax = deg2rad(random.uniform(-max_deg, max_deg))
    ay = deg2rad(random.uniform(-max_deg, max_deg))
    az = deg2rad(random.uniform(-max_deg, max_deg))
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(ax), -math.sin(ax)],
                   [0, math.sin(ax),  math.cos(ax)]])
    Ry = np.array([[ math.cos(ay), 0, math.sin(ay)],
                   [0,            1, 0],
                   [-math.sin(ay), 0, math.cos(ay)]])
    Rz = np.array([[math.cos(az), -math.sin(az), 0],
                   [math.sin(az),  math.cos(az), 0],
                   [0,            0,             1]])
    return Rz @ Ry @ Rx

def jitter(a: np.ndarray, std_frac: float) -> np.ndarray:
    mean, std, _ = standardize_trajectory(a)
    noise = np.random.normal(loc=0.0, scale=std_frac * std, size=a.shape)
    return a + noise

def translate(a: np.ndarray, std_frac: float) -> np.ndarray:
    _, std, _ = standardize_trajectory(a)
    t = np.random.normal(loc=0.0, scale=std_frac * std, size=(1, 3))
    return a + t

def scale_isotropic(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    s = random.uniform(lo, hi)
    return a * s

def rotate(a: np.ndarray, max_deg: float) -> np.ndarray:
    R = rand_rotation_matrix(max_deg)
    return (R @ a.T).T

def piecewise_time_warp(a: np.ndarray, n_knots: int = 3, strength: float = 0.1) -> np.ndarray:
    """
    단순 시간 왜곡: 인덱스를 [0,1]로 정규화하고 일부 지점(knots)에 작은 변형을 주어
    선형보간으로 재샘플링. 길이는 보존.
    """
    T = a.shape[0]
    if T < 4:
        return a
    # 원본 시간축
    t = np.linspace(0, 1, T)
    # 내부 knot 위치와 변형
    knots = np.linspace(0, 1, n_knots + 2)[1:-1]  # 0,1 제외
    displacements = np.random.uniform(-strength, strength, size=knots.shape)
    # 변형 프로파일 만들기(선형보간)
    ctrl_x = np.concatenate([[0.0], knots, [1.0]])
    ctrl_y = np.concatenate([[0.0], displacements, [0.0]])
    warp = np.interp(t, ctrl_x, ctrl_y)
    t_warped = np.clip(t + warp, 0, 1)

    # 각 좌표 보간으로 재샘플링
    out = np.zeros_like(a)
    for d in range(3):
        out[:, d] = np.interp(t, t_warped, a[:, d])
    return out

def pipeline_clean(a: np.ndarray) -> List[np.ndarray]:
    """클린 샘플: 약한 노이즈 + 회전/스케일/이동 + 가벼운 시간왜곡"""
    outs = []
    for _ in range(N_AUG_PER_CLEAN):
        b = a.copy()
        if random.random() < 0.7:
            b = jitter(b, JITTER_STD_FRAC_CLEAN)
        if random.random() < 0.9:
            b = rotate(b, ROT_DEG_MAX_CLEAN)
        if random.random() < 0.7:
            b = scale_isotropic(b, *SCALE_RANGE)
        if random.random() < 0.7:
            b = translate(b, TRANSLATE_STD_FRAC)
        if random.random() < 0.6:
            b = piecewise_time_warp(b, n_knots=3, strength=0.08)
        outs.append(b)
    return outs

def pipeline_noisy(a: np.ndarray) -> List[np.ndarray]:
    """
    노이즈 샘플: 먼저 스무딩/약한 클리핑으로 정제 후
    (1) 정제본 그대로
    (2) 정제본 + 미세 노이즈
    (3) 정제본 + 회전
    (4) 정제본 + 시간왜곡
    (5) 정제본 + 스케일/이동
    """
    outs = []
    # (1) 스무딩 + 약한 이상치 클리핑(IQR 기반)
    b = moving_average(a, SMOOTH_WINDOW)
    # IQR 클리핑
    q1, q3 = np.percentile(b, [25, 75], axis=0)
    iqr = q3 - q1 + 1e-9
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    b = np.clip(b, lo, hi)

    variants = []
    variants.append(b)  # (1)

    # (2) 미세 노이즈
    variants.append(jitter(b, JITTER_STD_FRAC_NOISY))

    # (3) 회전
    variants.append(rotate(b, ROT_DEG_MAX_NOISY))

    # (4) 시간왜곡(약하게)
    variants.append(piecewise_time_warp(b, n_knots=2, strength=0.06))

    # (5) 스케일 + 이동
    variants.append(translate(scale_isotropic(b, *SCALE_RANGE), TRANSLATE_STD_FRAC))

    # 필요 개수만큼 반환 (기본 5개)
    return variants[:N_AUG_PER_NOISY]


def main():
    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)

    root = Path(__file__).resolve().parent
    src_root = root / SRC_DIR
    dst_root = root / OUT_DIR
    dst_root.mkdir(exist_ok=True)

    total = 0
    for class_dir in sorted([p for p in src_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        category = class_dir.name
        for txt in sorted(class_dir.glob("*.txt"), key=lambda p: int(p.stem)):
            arr = read_xyz_txt(txt)  # (T,3)
            if arr.shape[0] < MIN_POINTS:
                print(f"[SKIP] {txt} (too few points: {arr.shape[0]})")
                continue

            stem_num = int(txt.stem) if txt.stem.isdigit() else -1
            noisy = is_noisy_file(category, stem_num)

            if noisy:
                gen_list = pipeline_noisy(arr)
            else:
                gen_list = pipeline_clean(arr)

            # 저장
            for k, aug in enumerate(gen_list, start=1):
                out_path = dst_root / category / f"{txt.stem}__aug{k:03d}.txt"
                write_xyz_txt(out_path, aug)

            total += len(gen_list)
            print(f"[OK] {txt.relative_to(root)} -> {len(gen_list)} aug (noisy={noisy})")

    print(f"\n■■■ Augmentation done. Generated {total} files under {dst_root}. ■■■")

if __name__ == "__main__":
    main()

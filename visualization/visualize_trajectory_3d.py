#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from ml_project import DataLoader, DataAugmentation

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def create_trajectory_3d_visualization():
    """시계열 궤적으로 증강 데이터를 3D 그래프로 시각화"""
    print("\n" + "=" * 80)
    print("3D 궤적 시각화 (Trajectory Visualization)")
    print("=" * 80)
    print("\n각 카테고리의 센서 데이터를 3D 궤적으로 표현합니다.")
    print("X축: 시간 순서, Y축: 데이터 값, Z축: Rolling Mean")
    
    base_path = str(Path(__file__).parent)
    loader = DataLoader(base_path)
    X, y = loader.load_data()
    
    viz_path = Path(base_path) / 'visualizations'
    viz_path.mkdir(exist_ok=True)
    
    categories = loader.categories
    
    # ========================================================================
    # 1. 각 카테고리별 원본 데이터 궤적 시각화
    # ========================================================================
    print("\n[1/2] 원본 데이터 궤적 시각화 생성 중...")
    
    fig = plt.figure(figsize=(18, 12))
    
    for idx, category in enumerate(categories):
        ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
        
        category_indices = [i for i, label in enumerate(y) if label == category]
        
        # 색상 팔레트
        colors = plt.cm.viridis(numpy.linspace(0, 1, len(category_indices)))
        
        print(f"  처리 중: {category} ({len(category_indices)}개 샘플)")
        
        for sample_idx, data_idx in enumerate(category_indices):
            sample = X[data_idx]
            
            # 데이터 서브샘플링 (모든 포인트를 사용하면 너무 복잡함)
            step = max(1, len(sample) // 500)
            sample_sub = sample[::step]
            
            # 3D 좌표 구성
            time_points = numpy.arange(len(sample_sub))  # X축: 시간
            values = sample_sub  # Y축: 데이터 값
            
            # Z축: Rolling Mean (window=5)
            rolling_mean = numpy.array([
                numpy.mean(sample[max(0, i*step-5):i*step+1]) 
                for i in range(len(sample_sub))
            ])
            
            # 3D 라인으로 궤적 그리기
            ax.plot(time_points, values, rolling_mean, 
                   color=colors[sample_idx], linewidth=2, alpha=0.8, label=f'Sample {sample_idx+1}')
            
            # 시작점과 끝점 표시
            ax.scatter(time_points[0], values[0], rolling_mean[0], 
                      color=colors[sample_idx], s=100, marker='o', edgecolors='black', linewidth=1.5)
            ax.scatter(time_points[-1], values[-1], rolling_mean[-1], 
                      color=colors[sample_idx], s=100, marker='s', edgecolors='black', linewidth=1.5)
        
        ax.set_xlabel('Time Index', fontsize=9, fontweight='bold')
        ax.set_ylabel('Value', fontsize=9, fontweight='bold')
        ax.set_zlabel('Rolling Mean', fontsize=9, fontweight='bold')
        ax.set_title(f'{category.upper()}\n(Original Data Trajectory)', fontsize=10, fontweight='bold')
        ax.view_init(elev=20, azim=45)
        ax.legend(fontsize=7, loc='upper left', ncol=2)
    
    plt.tight_layout()
    plt.savefig(str(viz_path / 'trajectory_3D_01_original_data.png'), dpi=300, bbox_inches='tight')
    print("  ✓ 저장: trajectory_3D_01_original_data.png")
    plt.close()
    
    # ========================================================================
    # 2. 원본 vs 증강 데이터 궤적 비교
    # ========================================================================
    print("\n[2/2] 원본 vs 증강 데이터 궤적 비교 생성 중...")
    
    for category in categories:
        print(f"  처리 중: {category}")
        
        fig = plt.figure(figsize=(16, 12))
        
        category_indices = [i for i, label in enumerate(y) if label == category]
        
        for view_idx, (elev, azim, view_name) in enumerate([
            (20, 45, 'View 1'),
            (20, 135, 'View 2'),
            (60, 45, 'View 3 (Top)'),
            (5, 0, 'View 4 (Side)')
        ]):
            ax = fig.add_subplot(2, 2, view_idx + 1, projection='3d')
            
            # 각 샘플마다 원본과 증강 데이터를 함께 표시
            colors = plt.cm.tab20(numpy.linspace(0, 1, len(category_indices) * 2))
            
            for sample_idx, data_idx in enumerate(category_indices):
                sample = X[data_idx]
                augmented = DataAugmentation.augment_features(sample)
                
                # 서브샘플링
                step = max(1, len(sample) // 500)
                sample_sub = sample[::step]
                
                # Z축: Rolling Mean
                rolling_mean = numpy.array([
                    numpy.mean(sample[max(0, i*step-5):i*step+1]) 
                    for i in range(len(sample_sub))
                ])
                
                time_points = numpy.arange(len(sample_sub))
                
                # 원본 데이터 (실선)
                ax.plot(time_points, sample_sub, rolling_mean,
                       color=f'C{sample_idx}', linewidth=2, alpha=0.8, linestyle='-', label=f'Original {sample_idx+1}')
                ax.scatter(time_points[0], sample_sub[0], rolling_mean[0],
                          color=f'C{sample_idx}', s=80, marker='o', edgecolors='black', linewidth=1)
                
                # 증강 데이터 (점선)
                aug_step = max(1, len(augmented) // 500)
                aug_sub = augmented[::aug_step]
                
                aug_rolling_mean = numpy.array([
                    numpy.mean(augmented[max(0, i*aug_step-5):i*aug_step+1]) 
                    for i in range(len(aug_sub))
                ])
                
                aug_time_points = numpy.arange(len(aug_sub))
                
                ax.plot(aug_time_points, aug_sub, aug_rolling_mean,
                       color=f'C{sample_idx}', linewidth=2, alpha=0.5, linestyle='--', label=f'Augmented {sample_idx+1}')
                ax.scatter(aug_time_points[-1], aug_sub[-1], aug_rolling_mean[-1],
                          color=f'C{sample_idx}', s=80, marker='^', edgecolors='black', linewidth=1)
            
            ax.set_xlabel('Time Index', fontsize=9, fontweight='bold')
            ax.set_ylabel('Value', fontsize=9, fontweight='bold')
            ax.set_zlabel('Rolling Mean', fontsize=9, fontweight='bold')
            ax.set_title(f'{view_name} (elev={elev}°, azim={azim}°)', fontsize=10, fontweight='bold')
            ax.view_init(elev=elev, azim=azim)
            ax.legend(fontsize=7, loc='upper left', ncol=2)
        
        fig.suptitle(f'{category.upper()} - 3D 궤적 분석\n(원본 실선 vs 증강 점선, 다중 각도)', 
                    fontsize=13, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(str(viz_path / f'trajectory_3D_02_{category}_augmented_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"    ✓ 저장: trajectory_3D_02_{category}_augmented_comparison.png")
        plt.close()
    
    print("\n" + "=" * 80)
    print("✅ 모든 궤적 3D 시각화 완료!")
    print("=" * 80)
    print(f"\n📊 저장된 위치: {viz_path}")
    print("\n생성된 파일:")
    print("  1. trajectory_3D_01_original_data.png")
    print("     → 5개 카테고리 원본 데이터의 3D 궤적")
    print("     → 원(○): 시작점, 사각형(■): 끝점")
    print("\n  2. trajectory_3D_02_{category}_augmented_comparison.png (5개)")
    print("     → 각 카테고리별 원본 vs 증강 궤적 비교")
    print("     → 실선: 원본 데이터, 점선: 증강 데이터")
    print("     → 4가지 각도 (Isometric, 180°, Top, Side)")


def main():
    create_trajectory_3d_visualization()


if __name__ == "__main__":
    main()

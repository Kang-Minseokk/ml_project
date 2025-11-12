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


def main():
    print("\n" + "=" * 80)
    print("3D 증강 데이터 시각화")
    print("=" * 80)
    print("\n각 카테고리(circle, diagonal_left, diagonal_right, horizontal, vertical)")
    print("에 대해 원본 vs 증강 데이터를 3D 그래프로 표현합니다.")
    print("\n축: Mean(평균), Std Dev(표준편차), Max(최댓값)")
    print("색상: 파란색=원본, 주황색=증강")
    
    base_path = str(Path(__file__).parent)
    loader = DataLoader(base_path)
    X, y = loader.load_data()
    
    viz_path = Path(base_path) / 'visualizations'
    viz_path.mkdir(exist_ok=True)
    
    categories = loader.categories
    
    # ========================================================================
    # 1. 모든 카테고리 한 화면 비교
    # ========================================================================
    print("\n[1/2] 모든 카테고리 한 화면 비교 생성 중...")
    
    fig = plt.figure(figsize=(18, 10))
    
    for idx, category in enumerate(categories):
        ax = fig.add_subplot(1, 5, idx + 1, projection='3d')
        
        category_indices = [i for i, label in enumerate(y) if label == category]
        category_samples = [X[i] for i in category_indices]
        
        print(f"  처리 중: {category} ({len(category_samples)}개)")
        
        means, stds, maxs = [], [], []
        aug_means, aug_stds, aug_maxs = [], [], []
        
        for sample in category_samples:
            augmented = DataAugmentation.augment_features(sample)
            
            means.append(numpy.mean(sample))
            stds.append(numpy.std(sample))
            maxs.append(numpy.max(sample))
            
            aug_means.append(numpy.mean(augmented))
            aug_stds.append(numpy.std(augmented))
            aug_maxs.append(numpy.max(augmented))
        
        ax.scatter(means, stds, maxs, c='blue', s=100, alpha=0.7, 
                  edgecolors='black', linewidth=1.5, label='Original')
        ax.scatter(aug_means, aug_stds, aug_maxs, c='orange', 
                  s=100, alpha=0.7, marker='^', edgecolors='black', linewidth=1.5, label='Augmented')
        
        for om, os, omx, am, ast, amx in zip(means, stds, maxs, aug_means, aug_stds, aug_maxs):
            ax.plot([om, am], [os, ast], [omx, amx], 'k--', alpha=0.2, linewidth=1)
        
        ax.set_xlabel('Mean', fontsize=8)
        ax.set_ylabel('Std Dev', fontsize=8)
        ax.set_zlabel('Max', fontsize=8)
        ax.set_title(f'{category.upper()}\n({len(category_samples)} samples)', fontsize=10, fontweight='bold')
        ax.view_init(elev=20, azim=45)
        ax.legend(fontsize=7)
    
    plt.tight_layout()
    plt.savefig(str(viz_path / '3D_01_all_categories_comparison.png'), dpi=300, bbox_inches='tight')
    print("  ✓ 저장: 3D_01_all_categories_comparison.png")
    plt.close()
    
    # ========================================================================
    # 2. 각 카테고리별 4가지 각도 상세 분석
    # ========================================================================
    print("\n[2/2] 각 카테고리별 4가지 각도 상세 분석 생성 중...")
    
    for category in categories:
        print(f"  처리 중: {category}")
        
        fig = plt.figure(figsize=(14, 10))
        
        category_indices = [i for i, label in enumerate(y) if label == category]
        category_samples = [X[i] for i in category_indices]
        
        means, stds, maxs = [], [], []
        aug_means, aug_stds, aug_maxs = [], [], []
        
        for sample in category_samples:
            augmented = DataAugmentation.augment_features(sample)
            
            means.append(numpy.mean(sample))
            stds.append(numpy.std(sample))
            maxs.append(numpy.max(sample))
            
            aug_means.append(numpy.mean(augmented))
            aug_stds.append(numpy.std(augmented))
            aug_maxs.append(numpy.max(augmented))
        
        # 4가지 각도
        angles = [(20, 45), (20, 135), (60, 45), (5, 0)]
        angle_labels = ['View 1 (Isometric)', 'View 2 (180°)', 'View 3 (Top)', 'View 4 (Side)']
        
        for angle_idx, (elev, azim) in enumerate(angles):
            ax = fig.add_subplot(2, 2, angle_idx + 1, projection='3d')
            
            ax.scatter(means, stds, maxs, c='blue', s=120, alpha=0.8, 
                      edgecolors='darkblue', linewidth=1.5, label='Original')
            ax.scatter(aug_means, aug_stds, aug_maxs, c='orange', 
                      s=120, alpha=0.8, marker='^', edgecolors='darkorange', linewidth=1.5, label='Augmented')
            
            for om, os, omx, am, ast, amx in zip(means, stds, maxs, aug_means, aug_stds, aug_maxs):
                ax.plot([om, am], [os, ast], [omx, amx], 'k--', alpha=0.25, linewidth=1)
            
            ax.set_xlabel('Mean', fontsize=9)
            ax.set_ylabel('Std Dev', fontsize=9)
            ax.set_zlabel('Max Value', fontsize=9)
            ax.set_title(f'{angle_labels[angle_idx]}\n(elev={elev}°, azim={azim}°)', fontsize=10, fontweight='bold')
            ax.view_init(elev=elev, azim=azim)
            ax.legend(fontsize=9)
        
        fig.suptitle(f'{category.upper()} - 3D 데이터 증강 분석\n(원본 vs 증강, 다중 각도)', 
                    fontsize=13, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(str(viz_path / f'3D_02_{category}_multiangle.png'), dpi=300, bbox_inches='tight')
        print(f"    ✓ 저장: 3D_02_{category}_multiangle.png")
        plt.close()
    
    print("\n" + "=" * 80)
    print("✅ 모든 3D 시각화 완료!")
    print("=" * 80)
    print(f"\n📊 저장된 위치: {viz_path}")
    print("\n생성된 파일:")
    print("  1. 3D_01_all_categories_comparison.png")
    print("     → 5개 카테고리 한 화면 비교 (원본 vs 증강)")
    print("\n  2. 3D_02_{category}_multiangle.png (5개)")
    print("     → 각 카테고리별 4가지 각도 상세 분석")
    print("       - View 1: 정각도 (Isometric)")
    print("       - View 2: 180도 회전")
    print("       - View 3: 위에서 본 모양 (Top)")
    print("       - View 4: 옆에서 본 모양 (Side)")


if __name__ == "__main__":
    main()

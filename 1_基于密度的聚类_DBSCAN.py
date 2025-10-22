# -*- coding: utf-8 -*-
"""
基于密度的聚类算法 - DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
Author: 教学示例
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons, make_blobs
from matplotlib import rcParams

# 设置中文字体为微软雅黑
rcParams['font.sans-serif'] = ['Microsoft YaHei']
rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("基于密度的聚类算法 - DBSCAN 详解")
print("=" * 80)

print("\n【核心概念】")
print("1. 核心点(Core Point): 在半径ε内至少有MinPts个点")
print("2. 边界点(Border Point): 在某个核心点的ε邻域内，但自身不是核心点")
print("3. 噪声点(Noise Point): 既不是核心点也不是边界点")
print("\n【算法优势】")
print("✓ 可以发现任意形状的簇")
print("✓ 能够识别噪声点")
print("✓ 不需要预先指定簇的数量")
print("✓ 对离群点不敏感")

# ============================================================================
# 示例1: 月牙形数据集（展示DBSCAN处理非凸形状的能力）
# ============================================================================
print("\n" + "=" * 80)
print("【示例1】月牙形数据集 - 展示DBSCAN处理复杂形状的能力")
print("=" * 80)

# 生成月牙形数据
X_moons, y_moons = make_moons(n_samples=300, noise=0.1, random_state=42)

# 应用DBSCAN算法
# eps: 邻域半径
# min_samples: 成为核心点所需的最小邻居数
dbscan_moons = DBSCAN(eps=0.2, min_samples=5)
labels_moons = dbscan_moons.fit_predict(X_moons)

print(f"\n参数设置:")
print(f"  - eps (邻域半径): 0.2")
print(f"  - min_samples (最小样本数): 5")
print(f"\n聚类结果:")
print(f"  - 发现的簇数量: {len(set(labels_moons)) - (1 if -1 in labels_moons else 0)}")
print(f"  - 噪声点数量: {list(labels_moons).count(-1)}")
print(f"  - 核心样本数量: {len(dbscan_moons.core_sample_indices_)}")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 原始数据
axes[0].scatter(X_moons[:, 0], X_moons[:, 1], c='gray', s=50, alpha=0.6)
axes[0].set_title('原始数据（月牙形）', fontsize=14, fontweight='bold')
axes[0].set_xlabel('特征 1')
axes[0].set_ylabel('特征 2')
axes[0].grid(True, alpha=0.3)

# DBSCAN聚类结果
scatter = axes[1].scatter(X_moons[:, 0], X_moons[:, 1], 
                          c=labels_moons, cmap='viridis', 
                          s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
# 标记核心点
core_samples_mask = np.zeros_like(labels_moons, dtype=bool)
core_samples_mask[dbscan_moons.core_sample_indices_] = True
axes[1].scatter(X_moons[core_samples_mask, 0], X_moons[core_samples_mask, 1],
                marker='o', s=100, edgecolors='red', linewidth=2, facecolors='none',
                label='核心点')
axes[1].set_title('DBSCAN聚类结果', fontsize=14, fontweight='bold')
axes[1].set_xlabel('特征 1')
axes[1].set_ylabel('特征 2')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.colorbar(scatter, ax=axes[1], label='簇标签（-1表示噪声）')
plt.tight_layout()
plt.savefig('输出图片/1_DBSCAN_月牙形.png', dpi=300, bbox_inches='tight')
print("\n✓ 图片已保存: 输出图片/1_DBSCAN_月牙形.png")

# ============================================================================
# 示例2: 含噪声的blob数据集（展示DBSCAN处理噪声的能力）
# ============================================================================
print("\n" + "=" * 80)
print("【示例2】含噪声数据集 - 展示DBSCAN识别噪声的能力")
print("=" * 80)

# 生成含噪声的数据
np.random.seed(42)
X_blobs, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=42)
# 添加随机噪声点
noise = np.random.uniform(low=-8, high=8, size=(50, 2))
X_noisy = np.vstack([X_blobs, noise])

# 应用DBSCAN
dbscan_noisy = DBSCAN(eps=0.8, min_samples=5)
labels_noisy = dbscan_noisy.fit_predict(X_noisy)

print(f"\n参数设置:")
print(f"  - eps (邻域半径): 0.8")
print(f"  - min_samples (最小样本数): 5")
print(f"\n聚类结果:")
print(f"  - 发现的簇数量: {len(set(labels_noisy)) - (1 if -1 in labels_noisy else 0)}")
print(f"  - 噪声点数量: {list(labels_noisy).count(-1)}")
print(f"  - 总样本数量: {len(X_noisy)}")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 原始数据
axes[0].scatter(X_noisy[:, 0], X_noisy[:, 1], c='gray', s=50, alpha=0.6)
axes[0].set_title('原始数据（含噪声）', fontsize=14, fontweight='bold')
axes[0].set_xlabel('特征 1')
axes[0].set_ylabel('特征 2')
axes[0].grid(True, alpha=0.3)

# DBSCAN结果（噪声点用红色标记）
noise_mask = labels_noisy == -1
axes[1].scatter(X_noisy[~noise_mask, 0], X_noisy[~noise_mask, 1],
                c=labels_noisy[~noise_mask], cmap='viridis', 
                s=50, alpha=0.8, edgecolors='black', linewidth=0.5, label='簇')
axes[1].scatter(X_noisy[noise_mask, 0], X_noisy[noise_mask, 1],
                c='red', marker='x', s=100, linewidth=2, label='噪声点')
axes[1].set_title('DBSCAN聚类结果（红叉为噪声）', fontsize=14, fontweight='bold')
axes[1].set_xlabel('特征 1')
axes[1].set_ylabel('特征 2')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('输出图片/2_DBSCAN_噪声处理.png', dpi=300, bbox_inches='tight')
print("\n✓ 图片已保存: 输出图片/2_DBSCAN_噪声处理.png")

# ============================================================================
# 示例3: 参数对比（不同eps和min_samples的影响）
# ============================================================================
print("\n" + "=" * 80)
print("【示例3】参数影响分析")
print("=" * 80)

X_demo, _ = make_moons(n_samples=200, noise=0.1, random_state=42)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('DBSCAN参数对聚类结果的影响', fontsize=16, fontweight='bold')

params = [
    (0.1, 5, 'eps=0.1, min_samples=5\n(过小的邻域)'),
    (0.2, 5, 'eps=0.2, min_samples=5\n(适中参数)'),
    (0.4, 5, 'eps=0.4, min_samples=5\n(过大的邻域)'),
    (0.2, 3, 'eps=0.2, min_samples=3\n(较少最小样本)'),
    (0.2, 10, 'eps=0.2, min_samples=10\n(较多最小样本)'),
    (0.3, 7, 'eps=0.3, min_samples=7\n(另一组参数)'),
]

for idx, (eps, min_samples, title) in enumerate(params):
    row = idx // 3
    col = idx % 3
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_demo)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    axes[row, col].scatter(X_demo[:, 0], X_demo[:, 1], 
                           c=labels, cmap='viridis', s=30, alpha=0.6)
    axes[row, col].set_title(f'{title}\n簇数:{n_clusters}, 噪声:{n_noise}', 
                             fontsize=10)
    axes[row, col].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('输出图片/3_DBSCAN_参数对比.png', dpi=300, bbox_inches='tight')
print("\n✓ 图片已保存: 输出图片/3_DBSCAN_参数对比.png")

print("\n" + "=" * 80)
print("【总结】DBSCAN算法特点")
print("=" * 80)
print("✓ 适用场景: 数据簇形状不规则、含有噪声、密度不均匀")
print("✓ 参数选择: eps和min_samples需要根据数据特点调整")
print("✓ 时间复杂度: O(n log n) (使用空间索引)")
print("=" * 80)

# 清理matplotlib资源
plt.close('all')


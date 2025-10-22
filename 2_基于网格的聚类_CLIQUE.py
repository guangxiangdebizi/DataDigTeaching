# -*- coding: utf-8 -*-
"""
基于网格的聚类算法 - CLIQUE (CLustering In QUEst)
Author: 教学示例
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from sklearn.datasets import make_blobs, make_moons
from matplotlib import rcParams
from scipy.spatial.distance import cdist

# 设置中文字体为微软雅黑
rcParams['font.sans-serif'] = ['Microsoft YaHei']
rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("基于网格的聚类算法 - CLIQUE 详解")
print("=" * 80)

print("\n【核心概念】")
print("1. 网格划分: 将数据空间划分为网格单元")
print("2. 密集单元: 包含的数据点数量超过阈值的网格单元")
print("3. 簇形成: 连接相邻的密集单元形成簇")
print("\n【算法优势】")
print("✓ 时间复杂度低，与数据点数量呈线性关系")
print("✓ 能够处理大规模数据集")
print("✓ 对参数不敏感")
print("✓ 可以发现任意形状的簇")
print("✓ 适合高维数据")

# ============================================================================
# 简化的CLIQUE算法实现
# ============================================================================
class SimpleCLIQUE:
    """简化的CLIQUE算法实现（用于教学演示）"""
    
    def __init__(self, n_grids=10, density_threshold=5):
        """
        参数:
            n_grids: 每个维度划分的网格数量
            density_threshold: 密集单元的最小点数阈值
        """
        self.n_grids = n_grids
        self.density_threshold = density_threshold
        self.grid_bounds_ = None
        self.dense_grids_ = None
        self.labels_ = None
        
    def fit_predict(self, X):
        """执行聚类并返回标签"""
        n_samples, n_features = X.shape
        
        # 步骤1: 计算网格边界
        self.grid_bounds_ = []
        for dim in range(n_features):
            min_val, max_val = X[:, dim].min(), X[:, dim].max()
            grid_width = (max_val - min_val) / self.n_grids
            self.grid_bounds_.append((min_val, max_val, grid_width))
        
        # 步骤2: 将每个点分配到网格
        grid_assignments = np.zeros((n_samples, n_features), dtype=int)
        for dim in range(n_features):
            min_val, max_val, grid_width = self.grid_bounds_[dim]
            grid_assignments[:, dim] = np.floor(
                (X[:, dim] - min_val) / grid_width
            ).astype(int)
            # 处理边界情况
            grid_assignments[grid_assignments[:, dim] >= self.n_grids, dim] = self.n_grids - 1
        
        # 步骤3: 统计每个网格的点数
        grid_counts = {}
        grid_points = {}  # 记录每个网格包含的点
        for i, grid_id in enumerate(grid_assignments):
            grid_key = tuple(grid_id)
            grid_counts[grid_key] = grid_counts.get(grid_key, 0) + 1
            if grid_key not in grid_points:
                grid_points[grid_key] = []
            grid_points[grid_key].append(i)
        
        # 步骤4: 识别密集网格
        self.dense_grids_ = {
            grid: count for grid, count in grid_counts.items() 
            if count >= self.density_threshold
        }
        
        print(f"\n网格统计:")
        print(f"  - 总网格数: {self.n_grids ** n_features}")
        print(f"  - 非空网格数: {len(grid_counts)}")
        print(f"  - 密集网格数: {len(self.dense_grids_)}")
        
        # 步骤5: 连接相邻的密集网格形成簇
        self.labels_ = -np.ones(n_samples, dtype=int)  # -1表示噪声
        cluster_id = 0
        visited = set()
        
        def get_neighbors(grid):
            """获取相邻的网格"""
            neighbors = []
            for dim in range(n_features):
                for delta in [-1, 1]:
                    neighbor = list(grid)
                    neighbor[dim] += delta
                    if 0 <= neighbor[dim] < self.n_grids:
                        neighbors.append(tuple(neighbor))
            return neighbors
        
        # 使用BFS连接相邻密集网格
        for grid in self.dense_grids_:
            if grid in visited:
                continue
            
            # 开始新簇
            queue = [grid]
            visited.add(grid)
            
            while queue:
                current_grid = queue.pop(0)
                # 标记该网格中的所有点
                if current_grid in grid_points:
                    for point_idx in grid_points[current_grid]:
                        self.labels_[point_idx] = cluster_id
                
                # 检查相邻网格
                for neighbor in get_neighbors(current_grid):
                    if neighbor in self.dense_grids_ and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            cluster_id += 1
        
        return self.labels_

# ============================================================================
# 示例1: 基本网格聚类
# ============================================================================
print("\n" + "=" * 80)
print("【示例1】基本网格聚类 - Blob数据集")
print("=" * 80)

# 生成数据
X_blobs, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.6, 
                        center_box=(-5, 5), random_state=42)

# 应用CLIQUE算法
clique = SimpleCLIQUE(n_grids=15, density_threshold=3)
labels_blobs = clique.fit_predict(X_blobs)

n_clusters = len(set(labels_blobs)) - (1 if -1 in labels_blobs else 0)
n_noise = list(labels_blobs).count(-1)

print(f"\n聚类结果:")
print(f"  - 发现的簇数量: {n_clusters}")
print(f"  - 噪声点数量: {n_noise}")

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 原始数据
axes[0].scatter(X_blobs[:, 0], X_blobs[:, 1], c='gray', s=50, alpha=0.6)
axes[0].set_title('原始数据', fontsize=14, fontweight='bold')
axes[0].set_xlabel('特征 1')
axes[0].set_ylabel('特征 2')
axes[0].grid(True, alpha=0.3)

# 网格划分
axes[1].scatter(X_blobs[:, 0], X_blobs[:, 1], c='gray', s=50, alpha=0.6)
# 绘制网格线
min_x, max_x, width_x = clique.grid_bounds_[0]
min_y, max_y, width_y = clique.grid_bounds_[1]
for i in range(clique.n_grids + 1):
    x = min_x + i * width_x
    axes[1].axvline(x, color='blue', alpha=0.3, linewidth=0.5)
    y = min_y + i * width_y
    axes[1].axhline(y, color='blue', alpha=0.3, linewidth=0.5)
# 标记密集网格
for grid in clique.dense_grids_:
    rect = patches.Rectangle(
        (min_x + grid[0] * width_x, min_y + grid[1] * width_y),
        width_x, width_y,
        linewidth=1, edgecolor='red', facecolor='yellow', alpha=0.3
    )
    axes[1].add_patch(rect)
axes[1].set_title(f'网格划分（黄色=密集网格）', fontsize=14, fontweight='bold')
axes[1].set_xlabel('特征 1')
axes[1].set_ylabel('特征 2')

# 聚类结果
scatter = axes[2].scatter(X_blobs[:, 0], X_blobs[:, 1], 
                          c=labels_blobs, cmap='viridis', 
                          s=50, alpha=0.8, edgecolors='black', linewidth=0.5)
axes[2].set_title('CLIQUE聚类结果', fontsize=14, fontweight='bold')
axes[2].set_xlabel('特征 1')
axes[2].set_ylabel('特征 2')
plt.colorbar(scatter, ax=axes[2], label='簇标签')

plt.tight_layout()
plt.savefig('输出图片/4_CLIQUE_基本聚类.png', dpi=300, bbox_inches='tight')
print("\n✓ 图片已保存: 输出图片/4_CLIQUE_基本聚类.png")

# ============================================================================
# 示例2: 不同网格大小的影响
# ============================================================================
print("\n" + "=" * 80)
print("【示例2】网格大小对聚类结果的影响")
print("=" * 80)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('网格数量对CLIQUE聚类的影响', fontsize=16, fontweight='bold')

grid_sizes = [5, 10, 15, 20, 25, 30]

for idx, n_grids in enumerate(grid_sizes):
    row = idx // 3
    col = idx % 3
    
    clique_test = SimpleCLIQUE(n_grids=n_grids, density_threshold=3)
    labels_test = clique_test.fit_predict(X_blobs)
    
    n_clusters = len(set(labels_test)) - (1 if -1 in labels_test else 0)
    n_noise = list(labels_test).count(-1)
    
    axes[row, col].scatter(X_blobs[:, 0], X_blobs[:, 1], 
                           c=labels_test, cmap='viridis', 
                           s=30, alpha=0.6, edgecolors='black', linewidth=0.3)
    axes[row, col].set_title(f'网格数={n_grids}×{n_grids}\n簇数:{n_clusters}, 噪声:{n_noise}', 
                             fontsize=10)
    axes[row, col].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('输出图片/5_CLIQUE_网格对比.png', dpi=300, bbox_inches='tight')
print("\n✓ 图片已保存: 输出图片/5_CLIQUE_网格对比.png")

# ============================================================================
# 示例3: 处理复杂形状
# ============================================================================
print("\n" + "=" * 80)
print("【示例3】处理复杂形状数据")
print("=" * 80)

X_moons, _ = make_moons(n_samples=300, noise=0.08, random_state=42)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 原始数据
axes[0].scatter(X_moons[:, 0], X_moons[:, 1], c='gray', s=50, alpha=0.6)
axes[0].set_title('原始数据（月牙形）', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# 粗网格
clique_coarse = SimpleCLIQUE(n_grids=8, density_threshold=3)
labels_coarse = clique_coarse.fit_predict(X_moons)
n_clusters_coarse = len(set(labels_coarse)) - (1 if -1 in labels_coarse else 0)
axes[1].scatter(X_moons[:, 0], X_moons[:, 1], 
                c=labels_coarse, cmap='viridis', s=50, alpha=0.6)
axes[1].set_title(f'粗网格(8×8)\n簇数: {n_clusters_coarse}', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# 细网格
clique_fine = SimpleCLIQUE(n_grids=20, density_threshold=2)
labels_fine = clique_fine.fit_predict(X_moons)
n_clusters_fine = len(set(labels_fine)) - (1 if -1 in labels_fine else 0)
axes[2].scatter(X_moons[:, 0], X_moons[:, 1], 
                c=labels_fine, cmap='viridis', s=50, alpha=0.6)
axes[2].set_title(f'细网格(20×20)\n簇数: {n_clusters_fine}', fontsize=14, fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('输出图片/6_CLIQUE_复杂形状.png', dpi=300, bbox_inches='tight')
print("\n✓ 图片已保存: 输出图片/6_CLIQUE_复杂形状.png")

print("\n" + "=" * 80)
print("【总结】CLIQUE算法特点")
print("=" * 80)
print("✓ 适用场景: 大规模数据集、高维数据")
print("✓ 优势: 时间复杂度O(n)，处理速度快")
print("✓ 参数: 网格数量和密度阈值")
print("✓ 局限: 网格大小选择影响聚类质量，对于非均匀密度数据效果可能不佳")
print("=" * 80)

# 清理matplotlib资源
plt.close('all')


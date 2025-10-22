# -*- coding: utf-8 -*-
"""
三种聚类算法的综合对比分析
Author: 教学示例
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs, make_moons, make_circles
from matplotlib import rcParams
import time

# 设置中文字体为微软雅黑
rcParams['font.sans-serif'] = ['Microsoft YaHei']
rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("三种聚类算法综合对比")
print("=" * 80)

# 简化的CLIQUE实现（从前面的代码复制）
class SimpleCLIQUE:
    def __init__(self, n_grids=10, density_threshold=5):
        self.n_grids = n_grids
        self.density_threshold = density_threshold
        self.labels_ = None
        
    def fit_predict(self, X):
        n_samples, n_features = X.shape
        grid_bounds = []
        
        for dim in range(n_features):
            min_val, max_val = X[:, dim].min(), X[:, dim].max()
            grid_width = (max_val - min_val) / self.n_grids
            grid_bounds.append((min_val, max_val, grid_width))
        
        grid_assignments = np.zeros((n_samples, n_features), dtype=int)
        for dim in range(n_features):
            min_val, max_val, grid_width = grid_bounds[dim]
            grid_assignments[:, dim] = np.floor(
                (X[:, dim] - min_val) / grid_width
            ).astype(int)
            grid_assignments[grid_assignments[:, dim] >= self.n_grids, dim] = self.n_grids - 1
        
        grid_counts = {}
        grid_points = {}
        for i, grid_id in enumerate(grid_assignments):
            grid_key = tuple(grid_id)
            grid_counts[grid_key] = grid_counts.get(grid_key, 0) + 1
            if grid_key not in grid_points:
                grid_points[grid_key] = []
            grid_points[grid_key].append(i)
        
        dense_grids = {
            grid: count for grid, count in grid_counts.items() 
            if count >= self.density_threshold
        }
        
        self.labels_ = -np.ones(n_samples, dtype=int)
        cluster_id = 0
        visited = set()
        
        def get_neighbors(grid):
            neighbors = []
            for dim in range(n_features):
                for delta in [-1, 1]:
                    neighbor = list(grid)
                    neighbor[dim] += delta
                    if 0 <= neighbor[dim] < self.n_grids:
                        neighbors.append(tuple(neighbor))
            return neighbors
        
        for grid in dense_grids:
            if grid in visited:
                continue
            
            queue = [grid]
            visited.add(grid)
            
            while queue:
                current_grid = queue.pop(0)
                if current_grid in grid_points:
                    for point_idx in grid_points[current_grid]:
                        self.labels_[point_idx] = cluster_id
                
                for neighbor in get_neighbors(current_grid):
                    if neighbor in dense_grids and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            cluster_id += 1
        
        return self.labels_

# ============================================================================
# 生成不同类型的测试数据集
# ============================================================================
print("\n【生成测试数据集】")

np.random.seed(42)

# 数据集1: 球形簇（适合所有算法）
X1, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.6, 
                   center_box=(-5, 5), random_state=42)

# 数据集2: 月牙形（非凸形状）
X2, _ = make_moons(n_samples=300, noise=0.1, random_state=42)

# 数据集3: 环形簇
X3, _ = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)

# 数据集4: 不同密度的簇
X4a, _ = make_blobs(n_samples=200, centers=1, cluster_std=0.5, 
                    center_box=(0, 0), random_state=42)
X4b, _ = make_blobs(n_samples=100, centers=1, cluster_std=1.5, 
                    center_box=(5, 5), random_state=43)
X4 = np.vstack([X4a, X4b])

# 数据集5: 不同大小和形状的簇（适合GMM）
X5 = np.vstack([
    np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 100),
    np.random.multivariate_normal([5, 5], [[2, 1], [1, 2]], 80),
    np.random.multivariate_normal([0, 8], [[0.5, 0], [0, 3]], 60),
])

datasets = [
    (X1, "球形簇"),
    (X2, "月牙形"),
    (X3, "环形簇"),
    (X4, "不同密度"),
    (X5, "不同形状")
]

print("✓ 已生成5个测试数据集")

# ============================================================================
# 对比分析
# ============================================================================
fig = plt.figure(figsize=(20, 16))
fig.suptitle('三种聚类算法综合对比\n(DBSCAN | CLIQUE | GMM)', 
             fontsize=18, fontweight='bold', y=0.995)

for idx, (X, name) in enumerate(datasets):
    print(f"\n{'='*60}")
    print(f"数据集: {name}")
    print(f"{'='*60}")
    
    # 原始数据
    ax_original = plt.subplot(5, 4, idx*4 + 1)
    ax_original.scatter(X[:, 0], X[:, 1], c='gray', s=30, alpha=0.6)
    ax_original.set_title(f'{name}\n(原始数据)', fontsize=10, fontweight='bold')
    ax_original.grid(True, alpha=0.3)
    ax_original.set_xticks([])
    ax_original.set_yticks([])
    
    # DBSCAN
    start_time = time.time()
    if name == "月牙形":
        dbscan = DBSCAN(eps=0.2, min_samples=5)
    elif name == "环形簇":
        dbscan = DBSCAN(eps=0.15, min_samples=5)
    elif name == "不同密度":
        dbscan = DBSCAN(eps=0.5, min_samples=5)
    else:
        dbscan = DBSCAN(eps=0.5, min_samples=5)
    
    labels_dbscan = dbscan.fit_predict(X)
    time_dbscan = time.time() - start_time
    n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
    n_noise_dbscan = list(labels_dbscan).count(-1)
    
    ax_dbscan = plt.subplot(5, 4, idx*4 + 2)
    ax_dbscan.scatter(X[:, 0], X[:, 1], c=labels_dbscan, 
                      cmap='viridis', s=30, alpha=0.6)
    ax_dbscan.set_title(f'DBSCAN\n簇:{n_clusters_dbscan} 噪声:{n_noise_dbscan}\n{time_dbscan:.3f}秒', 
                        fontsize=9)
    ax_dbscan.grid(True, alpha=0.3)
    ax_dbscan.set_xticks([])
    ax_dbscan.set_yticks([])
    
    print(f"DBSCAN: 簇数={n_clusters_dbscan}, 噪声={n_noise_dbscan}, 时间={time_dbscan:.4f}秒")
    
    # CLIQUE
    start_time = time.time()
    if name == "月牙形":
        clique = SimpleCLIQUE(n_grids=20, density_threshold=2)
    elif name == "环形簇":
        clique = SimpleCLIQUE(n_grids=15, density_threshold=3)
    else:
        clique = SimpleCLIQUE(n_grids=15, density_threshold=3)
    
    labels_clique = clique.fit_predict(X)
    time_clique = time.time() - start_time
    n_clusters_clique = len(set(labels_clique)) - (1 if -1 in labels_clique else 0)
    n_noise_clique = list(labels_clique).count(-1)
    
    ax_clique = plt.subplot(5, 4, idx*4 + 3)
    ax_clique.scatter(X[:, 0], X[:, 1], c=labels_clique, 
                      cmap='viridis', s=30, alpha=0.6)
    ax_clique.set_title(f'CLIQUE\n簇:{n_clusters_clique} 噪声:{n_noise_clique}\n{time_clique:.3f}秒', 
                        fontsize=9)
    ax_clique.grid(True, alpha=0.3)
    ax_clique.set_xticks([])
    ax_clique.set_yticks([])
    
    print(f"CLIQUE: 簇数={n_clusters_clique}, 噪声={n_noise_clique}, 时间={time_clique:.4f}秒")
    
    # GMM
    start_time = time.time()
    if name == "球形簇" or name == "不同形状":
        n_components = 3
    elif name == "月牙形":
        n_components = 2
    elif name == "环形簇":
        n_components = 2
    else:
        n_components = 2
    
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', 
                          random_state=42, max_iter=100)
    labels_gmm = gmm.fit_predict(X)
    time_gmm = time.time() - start_time
    n_clusters_gmm = len(set(labels_gmm))
    
    ax_gmm = plt.subplot(5, 4, idx*4 + 4)
    ax_gmm.scatter(X[:, 0], X[:, 1], c=labels_gmm, 
                   cmap='viridis', s=30, alpha=0.6)
    ax_gmm.set_title(f'GMM\n簇:{n_clusters_gmm}\n{time_gmm:.3f}秒', 
                     fontsize=9)
    ax_gmm.grid(True, alpha=0.3)
    ax_gmm.set_xticks([])
    ax_gmm.set_yticks([])
    
    print(f"GMM: 簇数={n_clusters_gmm}, 时间={time_gmm:.4f}秒")

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('输出图片/11_综合对比.png', dpi=300, bbox_inches='tight')
print("\n✓ 图片已保存: 输出图片/11_综合对比.png")

# ============================================================================
# 算法特性对比表
# ============================================================================
print("\n" + "=" * 80)
print("【算法特性对比表】")
print("=" * 80)

comparison_data = """
┌─────────────────┬──────────────────┬──────────────────┬──────────────────┐
│     特性        │     DBSCAN       │     CLIQUE       │       GMM        │
├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ 聚类方式        │   基于密度       │   基于网格       │   基于模型       │
├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ 簇形状          │   任意形状       │   任意形状       │   椭圆形         │
├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ 需要指定簇数    │   否             │   否             │   是             │
├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ 噪声处理        │   ✓✓✓          │   ✓✓           │   ✓             │
├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ 时间复杂度      │   O(n log n)     │   O(n)           │   O(n*k*i)       │
│                 │   (使用索引)     │   (线性)         │   (k=簇数,i=迭代)│
├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ 空间复杂度      │   O(n)           │   O(网格数)      │   O(n*k)         │
├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ 主要参数        │   eps            │   网格大小       │   簇数量         │
│                 │   min_samples    │   密度阈值       │   协方差类型     │
├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ 高维数据        │   一般           │   ✓✓✓          │   一般           │
├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ 大规模数据      │   ✓✓           │   ✓✓✓          │   ✓             │
├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ 软/硬聚类       │   硬聚类         │   硬聚类         │   软聚类         │
├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ 适用场景        │ 任意形状簇       │ 大规模/高维      │ 椭圆簇/概率输出  │
│                 │ 含噪声数据       │ 需要快速处理     │ 统计建模         │
└─────────────────┴──────────────────┴──────────────────┴──────────────────┘
"""

print(comparison_data)

print("\n" + "=" * 80)
print("【应用场景建议】")
print("=" * 80)
print("\n1. DBSCAN - 基于密度的聚类")
print("   ✓ 数据簇形状不规则（如月牙形、S形）")
print("   ✓ 数据中含有明显的噪声点或离群点")
print("   ✓ 簇的密度相对均匀")
print("   ✓ 不知道簇的数量")
print("   应用: 地理数据分析、异常检测、图像分割\n")

print("2. CLIQUE - 基于网格的聚类")
print("   ✓ 大规模数据集（百万级以上）")
print("   ✓ 高维数据（可进行维度约减）")
print("   ✓ 需要快速处理")
print("   ✓ 内存受限的环境")
print("   应用: 大数据处理、流数据分析、预处理\n")

print("3. GMM - 基于模型的聚类")
print("   ✓ 数据呈椭圆形分布")
print("   ✓ 需要概率输出（软聚类）")
print("   ✓ 簇的大小和形状各异")
print("   ✓ 需要统计解释")
print("   应用: 图像分割、语音识别、推荐系统、生物信息学\n")

print("=" * 80)
print("【参数调优建议】")
print("=" * 80)
print("\nDBSCAN:")
print("  - eps: 可通过K-距离图确定（观察拐点）")
print("  - min_samples: 一般设为维度数+1，或根据噪声容忍度调整")

print("\nCLIQUE:")
print("  - n_grids: 根据数据规模和分布，一般10-30之间")
print("  - density_threshold: 根据期望的最小簇大小设定")

print("\nGMM:")
print("  - n_components: 使用BIC/AIC准则选择")
print("  - covariance_type: full(灵活)>tied>diag>spherical(简单)")
print("=" * 80)

# 清理matplotlib资源
plt.close('all')


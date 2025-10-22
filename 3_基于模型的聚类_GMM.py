# -*- coding: utf-8 -*-
"""
基于模型的聚类算法 - GMM (Gaussian Mixture Model) 高斯混合模型
Author: 教学示例
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs, make_moons
from matplotlib.patches import Ellipse
from matplotlib import rcParams
from scipy.stats import multivariate_normal

# 设置中文字体为微软雅黑
rcParams['font.sans-serif'] = ['Microsoft YaHei']
rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("基于模型的聚类算法 - GMM (高斯混合模型) 详解")
print("=" * 80)

print("\n【核心概念】")
print("1. 概率模型: 假设数据由多个高斯分布混合而成")
print("2. EM算法: 使用期望最大化算法估计参数")
print("   - E步(Expectation): 计算每个点属于各个簇的概率")
print("   - M步(Maximization): 更新模型参数（均值、协方差）")
print("3. 软聚类: 每个点属于各个簇都有一个概率值")
print("\n【算法优势】")
print("✓ 提供概率输出，不是硬分类")
print("✓ 可以捕捉簇的形状和方向（通过协方差矩阵）")
print("✓ 有统计理论支持")
print("✓ 可以用AIC/BIC准则选择最佳簇数")

# ============================================================================
# 辅助函数：绘制高斯分布椭圆
# ============================================================================
def draw_ellipse(position, covariance, ax, **kwargs):
    """绘制表示高斯分布的椭圆"""
    # 计算特征值和特征向量
    vals, vecs = np.linalg.eigh(covariance)
    # 确保特征值为正
    vals = np.sqrt(np.abs(vals))
    
    # 计算椭圆的角度和宽高
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * 2 * vals  # 2倍标准差
    
    # 绘制椭圆
    ellipse = Ellipse(position, width, height, angle=angle, **kwargs)
    ax.add_patch(ellipse)

# ============================================================================
# 示例1: 基本GMM聚类
# ============================================================================
print("\n" + "=" * 80)
print("【示例1】基本GMM聚类 - 展示概率分布")
print("=" * 80)

# 生成数据
np.random.seed(42)
X_blobs, y_true = make_blobs(n_samples=300, centers=3, 
                              cluster_std=[1.0, 1.5, 0.5],
                              center_box=(-5, 5), random_state=42)

# 应用GMM
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X_blobs)
labels_gmm = gmm.predict(X_blobs)
probs = gmm.predict_proba(X_blobs)

print(f"\n模型参数:")
print(f"  - 簇数量: {gmm.n_components}")
print(f"  - 协方差类型: {gmm.covariance_type}")
print(f"  - 收敛: {'是' if gmm.converged_ else '否'}")
print(f"  - 迭代次数: {gmm.n_iter_}")

print(f"\n每个高斯分量的权重:")
for i, weight in enumerate(gmm.weights_):
    print(f"  - 簇 {i}: {weight:.3f}")

print(f"\n示例数据点的归属概率:")
for i in range(5):
    print(f"  - 点 {i}: ", end="")
    for j in range(3):
        print(f"簇{j}={probs[i,j]:.3f} ", end="")
    print()

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 原始数据
axes[0].scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_true, 
                cmap='viridis', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
axes[0].set_title('原始数据（真实标签）', fontsize=14, fontweight='bold')
axes[0].set_xlabel('特征 1')
axes[0].set_ylabel('特征 2')
axes[0].grid(True, alpha=0.3)

# GMM聚类结果 + 高斯分布椭圆
scatter = axes[1].scatter(X_blobs[:, 0], X_blobs[:, 1], 
                          c=labels_gmm, cmap='viridis', 
                          s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
# 绘制高斯分布椭圆
colors = ['red', 'green', 'blue']
for i in range(gmm.n_components):
    draw_ellipse(gmm.means_[i], gmm.covariances_[i], axes[1],
                 edgecolor=colors[i], facecolor='none', linewidth=2, 
                 linestyle='--', label=f'高斯分量 {i}')
    axes[1].plot(gmm.means_[i, 0], gmm.means_[i, 1], 'o', 
                 color=colors[i], markersize=12, markeredgecolor='black')
axes[1].set_title('GMM聚类结果（虚线=2σ椭圆）', fontsize=14, fontweight='bold')
axes[1].set_xlabel('特征 1')
axes[1].set_ylabel('特征 2')
axes[1].legend(loc='best')
axes[1].grid(True, alpha=0.3)

# 概率热图（显示属于第一个簇的概率）
xx, yy = np.meshgrid(np.linspace(X_blobs[:, 0].min()-1, X_blobs[:, 0].max()+1, 100),
                     np.linspace(X_blobs[:, 1].min()-1, X_blobs[:, 1].max()+1, 100))
Z = gmm.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 0]
Z = Z.reshape(xx.shape)

contour = axes[2].contourf(xx, yy, Z, levels=20, cmap='RdYlBu_r', alpha=0.6)
axes[2].scatter(X_blobs[:, 0], X_blobs[:, 1], c=labels_gmm, 
                cmap='viridis', s=30, alpha=0.8, edgecolors='black', linewidth=0.5)
axes[2].set_title('簇0的归属概率分布', fontsize=14, fontweight='bold')
axes[2].set_xlabel('特征 1')
axes[2].set_ylabel('特征 2')
plt.colorbar(contour, ax=axes[2], label='属于簇0的概率')

plt.tight_layout()
plt.savefig('输出图片/7_GMM_基本聚类.png', dpi=300, bbox_inches='tight')
print("\n✓ 图片已保存: 输出图片/7_GMM_基本聚类.png")

# ============================================================================
# 示例2: 不同协方差类型的对比
# ============================================================================
print("\n" + "=" * 80)
print("【示例2】协方差类型对聚类结果的影响")
print("=" * 80)

# 生成椭圆形簇
np.random.seed(42)
X_ellipse = np.vstack([
    np.random.multivariate_normal([0, 0], [[2, 1.5], [1.5, 2]], 100),
    np.random.multivariate_normal([5, 5], [[1, -0.7], [-0.7, 1]], 100),
    np.random.multivariate_normal([0, 5], [[1.5, 0], [0, 0.5]], 100),
])

covariance_types = ['spherical', 'tied', 'diag', 'full']
descriptions = [
    'spherical\n(球形,所有簇共享同一方差)',
    'tied\n(所有簇共享协方差矩阵)',
    'diag\n(对角协方差,轴对齐椭圆)',
    'full\n(完整协方差,任意椭圆)'
]

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for idx, (cov_type, desc) in enumerate(zip(covariance_types, descriptions)):
    gmm_test = GaussianMixture(n_components=3, covariance_type=cov_type, 
                                random_state=42, max_iter=200)
    gmm_test.fit(X_ellipse)
    labels_test = gmm_test.predict(X_ellipse)
    
    axes[idx].scatter(X_ellipse[:, 0], X_ellipse[:, 1], 
                      c=labels_test, cmap='viridis', s=30, alpha=0.6)
    
    # 绘制高斯椭圆
    colors = ['red', 'green', 'blue']
    for i in range(3):
        if cov_type == 'spherical':
            # 球形协方差
            cov = np.eye(2) * gmm_test.covariances_[i]
        elif cov_type == 'tied':
            # 共享协方差
            cov = gmm_test.covariances_
        elif cov_type == 'diag':
            # 对角协方差
            cov = np.diag(gmm_test.covariances_[i])
        else:  # full
            cov = gmm_test.covariances_[i]
        
        draw_ellipse(gmm_test.means_[i], cov, axes[idx],
                     edgecolor=colors[i], facecolor='none', linewidth=2)
    
    axes[idx].set_title(f'协方差类型: {desc}', fontsize=12, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_xlabel('特征 1')
    axes[idx].set_ylabel('特征 2')

plt.tight_layout()
plt.savefig('输出图片/8_GMM_协方差对比.png', dpi=300, bbox_inches='tight')
print("\n✓ 图片已保存: 输出图片/8_GMM_协方差对比.png")

# ============================================================================
# 示例3: 使用BIC选择最佳簇数
# ============================================================================
print("\n" + "=" * 80)
print("【示例3】使用BIC准则选择最佳簇数")
print("=" * 80)

# BIC (Bayesian Information Criterion) - 越小越好
n_components_range = range(1, 10)
bic_scores = []
aic_scores = []

for n in n_components_range:
    gmm_test = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
    gmm_test.fit(X_blobs)
    bic_scores.append(gmm_test.bic(X_blobs))
    aic_scores.append(gmm_test.aic(X_blobs))

best_n_components_bic = n_components_range[np.argmin(bic_scores)]
best_n_components_aic = n_components_range[np.argmin(aic_scores)]

print(f"\nBIC最佳簇数: {best_n_components_bic}")
print(f"AIC最佳簇数: {best_n_components_aic}")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# BIC/AIC曲线
axes[0].plot(n_components_range, bic_scores, 'o-', linewidth=2, 
             markersize=8, label='BIC', color='blue')
axes[0].plot(n_components_range, aic_scores, 's-', linewidth=2, 
             markersize=8, label='AIC', color='red')
axes[0].axvline(best_n_components_bic, color='blue', linestyle='--', 
                alpha=0.5, label=f'BIC最佳: {best_n_components_bic}')
axes[0].axvline(best_n_components_aic, color='red', linestyle='--', 
                alpha=0.5, label=f'AIC最佳: {best_n_components_aic}')
axes[0].set_xlabel('簇数量', fontsize=12)
axes[0].set_ylabel('信息准则值', fontsize=12)
axes[0].set_title('模型选择：BIC vs AIC', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 不同簇数的聚类结果对比
test_ns = [2, 3, 5]
colors_list = ['Reds', 'Greens', 'Blues']

for idx, test_n in enumerate(test_ns):
    gmm_viz = GaussianMixture(n_components=test_n, random_state=42)
    labels_viz = gmm_viz.fit_predict(X_blobs)
    axes[1].scatter(X_blobs[labels_viz==idx, 0] if idx < test_n else [], 
                    X_blobs[labels_viz==idx, 1] if idx < test_n else [],
                    alpha=0.3, s=20)

# 使用最佳参数
gmm_best = GaussianMixture(n_components=best_n_components_bic, random_state=42)
labels_best = gmm_best.fit_predict(X_blobs)
scatter = axes[1].scatter(X_blobs[:, 0], X_blobs[:, 1], 
                          c=labels_best, cmap='viridis', 
                          s=50, alpha=0.8, edgecolors='black', linewidth=0.5)
axes[1].set_title(f'最佳簇数 = {best_n_components_bic}', 
                  fontsize=14, fontweight='bold')
axes[1].set_xlabel('特征 1')
axes[1].set_ylabel('特征 2')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('输出图片/9_GMM_模型选择.png', dpi=300, bbox_inches='tight')
print("\n✓ 图片已保存: 输出图片/9_GMM_模型选择.png")

# ============================================================================
# 示例4: GMM vs K-Means对比
# ============================================================================
print("\n" + "=" * 80)
print("【示例4】GMM与K-Means的对比")
print("=" * 80)

from sklearn.cluster import KMeans

# 生成具有不同形状和大小的簇
np.random.seed(42)
X_varied = np.vstack([
    np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 100),
    np.random.multivariate_normal([4, 4], [[3, 1], [1, 3]], 100),
    np.random.multivariate_normal([0, 6], [[0.5, 0], [0, 2]], 100),
])

# K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_kmeans = kmeans.fit_predict(X_varied)

# GMM
gmm_compare = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
labels_gmm_compare = gmm_compare.fit_predict(X_varied)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# K-Means结果
axes[0].scatter(X_varied[:, 0], X_varied[:, 1], 
                c=labels_kmeans, cmap='viridis', 
                s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
axes[0].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                marker='X', s=200, c='red', edgecolors='black', linewidth=2,
                label='簇中心')
axes[0].set_title('K-Means聚类\n(硬聚类,球形簇)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('特征 1')
axes[0].set_ylabel('特征 2')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# GMM结果
axes[1].scatter(X_varied[:, 0], X_varied[:, 1], 
                c=labels_gmm_compare, cmap='viridis', 
                s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
colors = ['red', 'green', 'blue']
for i in range(3):
    draw_ellipse(gmm_compare.means_[i], gmm_compare.covariances_[i], axes[1],
                 edgecolor=colors[i], facecolor='none', linewidth=2)
    axes[1].plot(gmm_compare.means_[i, 0], gmm_compare.means_[i, 1], 'o',
                 color=colors[i], markersize=12, markeredgecolor='black')
axes[1].set_title('GMM聚类\n(软聚类,椭圆簇)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('特征 1')
axes[1].set_ylabel('特征 2')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('输出图片/10_GMM_vs_KMeans.png', dpi=300, bbox_inches='tight')
print("\n✓ 图片已保存: 输出图片/10_GMM_vs_KMeans.png")

print("\n" + "=" * 80)
print("【总结】GMM算法特点")
print("=" * 80)
print("✓ 适用场景: 簇呈椭圆形分布、需要概率输出、簇大小不一")
print("✓ 优势: 软聚类、捕捉簇的形状和方向、有模型选择准则")
print("✓ 参数: 簇数量、协方差类型")
print("✓ 局限: 对初始值敏感、假设数据服从高斯分布")
print("✓ 与K-Means对比: GMM更灵活，可以处理不同形状和大小的簇")
print("=" * 80)

# 清理matplotlib资源
plt.close('all')


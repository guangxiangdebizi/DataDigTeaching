# -*- coding: utf-8 -*-
"""
使用真实数据集进行聚类分析
包含：Iris（鸢尾花）、Wine（葡萄酒）、Digits（手写数字）数据集
Author: 教学示例
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score
from matplotlib import rcParams
import pandas as pd

# 设置中文字体为微软雅黑
rcParams['font.sans-serif'] = ['Microsoft YaHei']
rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("真实数据集聚类分析")
print("=" * 80)

# ============================================================================
# 数据集1: Iris（鸢尾花数据集）- 经典的分类数据集
# ============================================================================
print("\n" + "=" * 80)
print("【数据集1】Iris 鸢尾花数据集")
print("=" * 80)

iris = load_iris()
X_iris = iris.data
y_iris = iris.target

print(f"\n数据集信息:")
print(f"  - 样本数量: {X_iris.shape[0]}")
print(f"  - 特征数量: {X_iris.shape[1]}")
print(f"  - 类别数量: {len(np.unique(y_iris))}")
print(f"  - 特征名称: {iris.feature_names}")
print(f"  - 类别名称: {iris.target_names}")

# 数据标准化
scaler = StandardScaler()
X_iris_scaled = scaler.fit_transform(X_iris)

# 使用PCA降维到2D用于可视化
pca_iris = PCA(n_components=2)
X_iris_2d = pca_iris.fit_transform(X_iris_scaled)

print(f"\nPCA降维信息:")
print(f"  - 前2个主成分解释的方差比: {pca_iris.explained_variance_ratio_}")
print(f"  - 累计方差比: {sum(pca_iris.explained_variance_ratio_):.2%}")

# 应用三种聚类算法
kmeans_iris = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_kmeans_iris = kmeans_iris.fit_predict(X_iris_scaled)

dbscan_iris = DBSCAN(eps=0.5, min_samples=5)
labels_dbscan_iris = dbscan_iris.fit_predict(X_iris_scaled)

gmm_iris = GaussianMixture(n_components=3, random_state=42)
labels_gmm_iris = gmm_iris.fit_predict(X_iris_scaled)

# 计算评估指标
print(f"\n聚类评估（与真实标签对比）:")
print(f"  K-Means ARI: {adjusted_rand_score(y_iris, labels_kmeans_iris):.3f}")
print(f"  DBSCAN ARI:  {adjusted_rand_score(y_iris, labels_dbscan_iris):.3f}")
print(f"  GMM ARI:     {adjusted_rand_score(y_iris, labels_gmm_iris):.3f}")

print(f"\n轮廓系数（Silhouette Score，越接近1越好）:")
print(f"  K-Means: {silhouette_score(X_iris_scaled, labels_kmeans_iris):.3f}")
if len(set(labels_dbscan_iris)) > 1:
    print(f"  DBSCAN:  {silhouette_score(X_iris_scaled, labels_dbscan_iris):.3f}")
print(f"  GMM:     {silhouette_score(X_iris_scaled, labels_gmm_iris):.3f}")

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Iris鸢尾花数据集聚类分析\n(PCA降维到2D可视化)', fontsize=16, fontweight='bold')

# 真实标签
scatter0 = axes[0, 0].scatter(X_iris_2d[:, 0], X_iris_2d[:, 1], 
                              c=y_iris, cmap='viridis', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
axes[0, 0].set_title('真实标签（Ground Truth）', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('第1主成分')
axes[0, 0].set_ylabel('第2主成分')
axes[0, 0].grid(True, alpha=0.3)
cbar0 = plt.colorbar(scatter0, ax=axes[0, 0])
cbar0.set_ticks([0, 1, 2])
cbar0.set_ticklabels(iris.target_names)

# K-Means
scatter1 = axes[0, 1].scatter(X_iris_2d[:, 0], X_iris_2d[:, 1], 
                              c=labels_kmeans_iris, cmap='viridis', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
axes[0, 1].set_title(f'K-Means (ARI={adjusted_rand_score(y_iris, labels_kmeans_iris):.3f})', 
                     fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('第1主成分')
axes[0, 1].set_ylabel('第2主成分')
axes[0, 1].grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=axes[0, 1])

# DBSCAN
scatter2 = axes[1, 0].scatter(X_iris_2d[:, 0], X_iris_2d[:, 1], 
                              c=labels_dbscan_iris, cmap='viridis', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
axes[1, 0].set_title(f'DBSCAN (ARI={adjusted_rand_score(y_iris, labels_dbscan_iris):.3f})', 
                     fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('第1主成分')
axes[1, 0].set_ylabel('第2主成分')
axes[1, 0].grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=axes[1, 0])

# GMM
scatter3 = axes[1, 1].scatter(X_iris_2d[:, 0], X_iris_2d[:, 1], 
                              c=labels_gmm_iris, cmap='viridis', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
axes[1, 1].set_title(f'GMM (ARI={adjusted_rand_score(y_iris, labels_gmm_iris):.3f})', 
                     fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('第1主成分')
axes[1, 1].set_ylabel('第2主成分')
axes[1, 1].grid(True, alpha=0.3)
plt.colorbar(scatter3, ax=axes[1, 1])

plt.tight_layout()
plt.savefig('输出图片/12_真实数据_Iris.png', dpi=300, bbox_inches='tight')
print("\n✓ 图片已保存: 输出图片/12_真实数据_Iris.png")

# ============================================================================
# 数据集2: Wine（葡萄酒数据集）- 化学成分分析
# ============================================================================
print("\n" + "=" * 80)
print("【数据集2】Wine 葡萄酒数据集")
print("=" * 80)

wine = load_wine()
X_wine = wine.data
y_wine = wine.target

print(f"\n数据集信息:")
print(f"  - 样本数量: {X_wine.shape[0]}")
print(f"  - 特征数量: {X_wine.shape[1]}")
print(f"  - 类别数量: {len(np.unique(y_wine))}")
print(f"  - 特征示例: {wine.feature_names[:5]}")

# 标准化和降维
X_wine_scaled = scaler.fit_transform(X_wine)
pca_wine = PCA(n_components=2)
X_wine_2d = pca_wine.fit_transform(X_wine_scaled)

print(f"\nPCA降维信息:")
print(f"  - 前2个主成分解释的方差比: {pca_wine.explained_variance_ratio_}")
print(f"  - 累计方差比: {sum(pca_wine.explained_variance_ratio_):.2%}")

# 聚类
kmeans_wine = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_kmeans_wine = kmeans_wine.fit_predict(X_wine_scaled)

dbscan_wine = DBSCAN(eps=3, min_samples=5)
labels_dbscan_wine = dbscan_wine.fit_predict(X_wine_scaled)

gmm_wine = GaussianMixture(n_components=3, random_state=42)
labels_gmm_wine = gmm_wine.fit_predict(X_wine_scaled)

print(f"\n聚类评估（与真实标签对比）:")
print(f"  K-Means ARI: {adjusted_rand_score(y_wine, labels_kmeans_wine):.3f}")
print(f"  DBSCAN ARI:  {adjusted_rand_score(y_wine, labels_dbscan_wine):.3f}")
print(f"  GMM ARI:     {adjusted_rand_score(y_wine, labels_gmm_wine):.3f}")

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Wine葡萄酒数据集聚类分析\n(PCA降维到2D可视化)', fontsize=16, fontweight='bold')

# 真实标签
scatter0 = axes[0, 0].scatter(X_wine_2d[:, 0], X_wine_2d[:, 1], 
                              c=y_wine, cmap='viridis', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
axes[0, 0].set_title('真实标签（3种葡萄酒类型）', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('第1主成分')
axes[0, 0].set_ylabel('第2主成分')
axes[0, 0].grid(True, alpha=0.3)
plt.colorbar(scatter0, ax=axes[0, 0])

# K-Means
scatter1 = axes[0, 1].scatter(X_wine_2d[:, 0], X_wine_2d[:, 1], 
                              c=labels_kmeans_wine, cmap='viridis', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
axes[0, 1].set_title(f'K-Means (ARI={adjusted_rand_score(y_wine, labels_kmeans_wine):.3f})', 
                     fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('第1主成分')
axes[0, 1].set_ylabel('第2主成分')
axes[0, 1].grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=axes[0, 1])

# DBSCAN
scatter2 = axes[1, 0].scatter(X_wine_2d[:, 0], X_wine_2d[:, 1], 
                              c=labels_dbscan_wine, cmap='viridis', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
axes[1, 0].set_title(f'DBSCAN (ARI={adjusted_rand_score(y_wine, labels_dbscan_wine):.3f})', 
                     fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('第1主成分')
axes[1, 0].set_ylabel('第2主成分')
axes[1, 0].grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=axes[1, 0])

# GMM
scatter3 = axes[1, 1].scatter(X_wine_2d[:, 0], X_wine_2d[:, 1], 
                              c=labels_gmm_wine, cmap='viridis', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
axes[1, 1].set_title(f'GMM (ARI={adjusted_rand_score(y_wine, labels_gmm_wine):.3f})', 
                     fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('第1主成分')
axes[1, 1].set_ylabel('第2主成分')
axes[1, 1].grid(True, alpha=0.3)
plt.colorbar(scatter3, ax=axes[1, 1])

plt.tight_layout()
plt.savefig('输出图片/13_真实数据_Wine.png', dpi=300, bbox_inches='tight')
print("\n✓ 图片已保存: 输出图片/13_真实数据_Wine.png")

# ============================================================================
# 数据集3: Digits（手写数字数据集）- 图像数据
# ============================================================================
print("\n" + "=" * 80)
print("【数据集3】Digits 手写数字数据集")
print("=" * 80)

digits = load_digits()
X_digits = digits.data
y_digits = digits.target

print(f"\n数据集信息:")
print(f"  - 样本数量: {X_digits.shape[0]}")
print(f"  - 特征数量: {X_digits.shape[1]} (8×8像素的图像)")
print(f"  - 类别数量: {len(np.unique(y_digits))} (数字0-9)")

# 由于数字有10个类别，我们只聚类前3个数字作为演示
mask = y_digits < 3
X_digits_subset = X_digits[mask]
y_digits_subset = y_digits[mask]

print(f"\n使用子集进行演示:")
print(f"  - 只选择数字0,1,2")
print(f"  - 样本数量: {X_digits_subset.shape[0]}")

# 标准化和降维
X_digits_scaled = scaler.fit_transform(X_digits_subset)
pca_digits = PCA(n_components=2)
X_digits_2d = pca_digits.fit_transform(X_digits_scaled)

print(f"\nPCA降维信息:")
print(f"  - 前2个主成分解释的方差比: {pca_digits.explained_variance_ratio_}")
print(f"  - 累计方差比: {sum(pca_digits.explained_variance_ratio_):.2%}")

# 聚类
kmeans_digits = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_kmeans_digits = kmeans_digits.fit_predict(X_digits_scaled)

dbscan_digits = DBSCAN(eps=5, min_samples=5)
labels_dbscan_digits = dbscan_digits.fit_predict(X_digits_scaled)

gmm_digits = GaussianMixture(n_components=3, random_state=42)
labels_gmm_digits = gmm_digits.fit_predict(X_digits_scaled)

print(f"\n聚类评估（与真实标签对比）:")
print(f"  K-Means ARI: {adjusted_rand_score(y_digits_subset, labels_kmeans_digits):.3f}")
print(f"  DBSCAN ARI:  {adjusted_rand_score(y_digits_subset, labels_dbscan_digits):.3f}")
print(f"  GMM ARI:     {adjusted_rand_score(y_digits_subset, labels_gmm_digits):.3f}")

# 可视化
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

fig.suptitle('Digits手写数字数据集聚类分析\n(只使用数字0,1,2)', fontsize=16, fontweight='bold')

# 展示一些原始图像
ax_images = fig.add_subplot(gs[0, 0])
for i in range(10):
    ax = plt.subplot(gs[0, 0])
    if i == 0:
        # 创建一个小图展示原始图像示例
        sample_images = []
        for digit in range(3):
            idx = np.where(y_digits_subset == digit)[0][0]
            sample_images.append(digits.images[mask][idx])
        
        combined = np.hstack(sample_images)
        ax.imshow(combined, cmap='gray')
        ax.set_title('原始图像示例\n(数字0, 1, 2)', fontsize=10, fontweight='bold')
        ax.axis('off')
        break

# 真实标签
ax1 = fig.add_subplot(gs[0, 1])
scatter0 = ax1.scatter(X_digits_2d[:, 0], X_digits_2d[:, 1], 
                       c=y_digits_subset, cmap='viridis', s=30, alpha=0.6, edgecolors='black', linewidth=0.5)
ax1.set_title('真实标签', fontsize=12, fontweight='bold')
ax1.set_xlabel('第1主成分')
ax1.set_ylabel('第2主成分')
ax1.grid(True, alpha=0.3)
plt.colorbar(scatter0, ax=ax1)

# 数据统计
ax_stat = fig.add_subplot(gs[0, 2])
ax_stat.axis('off')
stats_text = f"""
数据集统计:

总样本数: {X_digits_subset.shape[0]}
  - 数字0: {sum(y_digits_subset==0)}个
  - 数字1: {sum(y_digits_subset==1)}个  
  - 数字2: {sum(y_digits_subset==2)}个

特征维度: {X_digits.shape[1]}
(8×8像素图像)

降维后保留信息:
  {sum(pca_digits.explained_variance_ratio_):.1%}
"""
ax_stat.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# K-Means
ax2 = fig.add_subplot(gs[1, 0])
scatter1 = ax2.scatter(X_digits_2d[:, 0], X_digits_2d[:, 1], 
                       c=labels_kmeans_digits, cmap='viridis', s=30, alpha=0.6, edgecolors='black', linewidth=0.5)
ax2.set_title(f'K-Means\nARI={adjusted_rand_score(y_digits_subset, labels_kmeans_digits):.3f}', 
              fontsize=12, fontweight='bold')
ax2.set_xlabel('第1主成分')
ax2.set_ylabel('第2主成分')
ax2.grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=ax2)

# DBSCAN
ax3 = fig.add_subplot(gs[1, 1])
scatter2 = ax3.scatter(X_digits_2d[:, 0], X_digits_2d[:, 1], 
                       c=labels_dbscan_digits, cmap='viridis', s=30, alpha=0.6, edgecolors='black', linewidth=0.5)
ax3.set_title(f'DBSCAN\nARI={adjusted_rand_score(y_digits_subset, labels_dbscan_digits):.3f}', 
              fontsize=12, fontweight='bold')
ax3.set_xlabel('第1主成分')
ax3.set_ylabel('第2主成分')
ax3.grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=ax3)

# GMM
ax4 = fig.add_subplot(gs[1, 2])
scatter3 = ax4.scatter(X_digits_2d[:, 0], X_digits_2d[:, 1], 
                       c=labels_gmm_digits, cmap='viridis', s=30, alpha=0.6, edgecolors='black', linewidth=0.5)
ax4.set_title(f'GMM\nARI={adjusted_rand_score(y_digits_subset, labels_gmm_digits):.3f}', 
              fontsize=12, fontweight='bold')
ax4.set_xlabel('第1主成分')
ax4.set_ylabel('第2主成分')
ax4.grid(True, alpha=0.3)
plt.colorbar(scatter3, ax=ax4)

# 评估指标对比图
ax5 = fig.add_subplot(gs[2, :])
datasets_names = ['Iris', 'Wine', 'Digits']
ari_scores = {
    'K-Means': [
        adjusted_rand_score(y_iris, labels_kmeans_iris),
        adjusted_rand_score(y_wine, labels_kmeans_wine),
        adjusted_rand_score(y_digits_subset, labels_kmeans_digits)
    ],
    'DBSCAN': [
        adjusted_rand_score(y_iris, labels_dbscan_iris),
        adjusted_rand_score(y_wine, labels_dbscan_wine),
        adjusted_rand_score(y_digits_subset, labels_dbscan_digits)
    ],
    'GMM': [
        adjusted_rand_score(y_iris, labels_gmm_iris),
        adjusted_rand_score(y_wine, labels_gmm_wine),
        adjusted_rand_score(y_digits_subset, labels_gmm_digits)
    ]
}

x = np.arange(len(datasets_names))
width = 0.25

bars1 = ax5.bar(x - width, ari_scores['K-Means'], width, label='K-Means', alpha=0.8)
bars2 = ax5.bar(x, ari_scores['DBSCAN'], width, label='DBSCAN', alpha=0.8)
bars3 = ax5.bar(x + width, ari_scores['GMM'], width, label='GMM', alpha=0.8)

ax5.set_ylabel('ARI得分（越高越好）', fontsize=12)
ax5.set_title('三个真实数据集上的聚类性能对比\n(ARI: Adjusted Rand Index)', fontsize=13, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(datasets_names)
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')
ax5.set_ylim([0, 1])

# 在柱状图上添加数值标签
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.savefig('输出图片/14_真实数据_Digits.png', dpi=300, bbox_inches='tight')
print("\n✓ 图片已保存: 输出图片/14_真实数据_Digits.png")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "=" * 80)
print("【真实数据集聚类总结】")
print("=" * 80)

summary_df = pd.DataFrame({
    '数据集': ['Iris', 'Wine', 'Digits(0-2)'],
    '样本数': [X_iris.shape[0], X_wine.shape[0], X_digits_subset.shape[0]],
    '特征数': [X_iris.shape[1], X_wine.shape[1], X_digits.shape[1]],
    '类别数': [3, 3, 3],
    'K-Means ARI': [f"{adjusted_rand_score(y_iris, labels_kmeans_iris):.3f}",
                     f"{adjusted_rand_score(y_wine, labels_kmeans_wine):.3f}",
                     f"{adjusted_rand_score(y_digits_subset, labels_kmeans_digits):.3f}"],
    'DBSCAN ARI': [f"{adjusted_rand_score(y_iris, labels_dbscan_iris):.3f}",
                    f"{adjusted_rand_score(y_wine, labels_dbscan_wine):.3f}",
                    f"{adjusted_rand_score(y_digits_subset, labels_dbscan_digits):.3f}"],
    'GMM ARI': [f"{adjusted_rand_score(y_iris, labels_gmm_iris):.3f}",
                 f"{adjusted_rand_score(y_wine, labels_gmm_wine):.3f}",
                 f"{adjusted_rand_score(y_digits_subset, labels_gmm_digits):.3f}"]
})

print("\n" + summary_df.to_string(index=False))

print("\n\n【关键观察】")
print("1. Iris数据集: 三种算法都表现良好，数据结构清晰")
print("2. Wine数据集: GMM表现最佳，可能因为簇呈椭圆分布")
print("3. Digits数据集: 图像数据高维复杂，需要更多特征")
print("\n【ARI指标说明】")
print("  - ARI = 1: 完美匹配真实标签")
print("  - ARI = 0: 随机聚类")
print("  - ARI < 0: 比随机还差")
print("=" * 80)

# 清理matplotlib资源
plt.close('all')


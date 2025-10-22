# -*- coding: utf-8 -*-
"""
运行所有聚类算法示例
Author: 教学示例
"""

import os
import sys
from matplotlib import rcParams
import traceback

# 设置中文字体为微软雅黑
rcParams['font.sans-serif'] = ['Microsoft YaHei']
rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("开始运行所有聚类算法教学示例")
print("=" * 80)

# 创建输出目录
if not os.path.exists('输出图片'):
    os.makedirs('输出图片')
    print("✓ 已创建输出目录: 输出图片/")

scripts = [
    "1_基于密度的聚类_DBSCAN.py",
    "2_基于网格的聚类_CLIQUE.py",
    "3_基于模型的聚类_GMM.py",
    "4_综合对比分析.py",
    "5_真实数据集示例.py"
]

successful_scripts = []
failed_scripts = []

for i, script in enumerate(scripts, 1):
    print(f"\n{'='*80}")
    print(f"[{i}/{len(scripts)}] 正在运行: {script}")
    print(f"{'='*80}\n")
    
    try:
        # 动态导入并执行
        if os.path.exists(script):
            with open(script, 'r', encoding='utf-8') as f:
                code = f.read()
                exec(code, {'__name__': '__main__'})  # 设置 __name__ 为 '__main__'
            print(f"\n✓ {script} 运行完成！")
            successful_scripts.append(script)
        else:
            print(f"\n✗ {script} 文件不存在！")
            failed_scripts.append(script)
    except Exception as e:
        print(f"\n✗ {script} 运行出错: {str(e)}")
        traceback.print_exc()
        failed_scripts.append(script)

print("\n" + "=" * 80)
print("运行总结")
print("=" * 80)
print(f"✓ 成功运行: {len(successful_scripts)} 个脚本")
for script in successful_scripts:
    print(f"    - {script}")
    
if failed_scripts:
    print(f"\n✗ 失败的脚本: {len(failed_scripts)} 个")
    for script in failed_scripts:
        print(f"    - {script}")
else:
    print("\n🎉 所有示例都运行成功！")

print("\n" + "=" * 80)
print("📊 生成的图片保存在 '输出图片/' 目录下")
print("\n包含以下文件:")
print("  1. 1_DBSCAN_月牙形.png")
print("  2. 2_DBSCAN_噪声处理.png")
print("  3. 3_DBSCAN_参数对比.png")
print("  4. 4_CLIQUE_基本聚类.png")
print("  5. 5_CLIQUE_网格对比.png")
print("  6. 6_CLIQUE_复杂形状.png")
print("  7. 7_GMM_基本聚类.png")
print("  8. 8_GMM_协方差对比.png")
print("  9. 9_GMM_模型选择.png")
print("  10. 10_GMM_vs_KMeans.png")
print("  11. 11_综合对比.png")
print("  12. 12_真实数据_Iris.png")
print("  13. 13_真实数据_Wine.png")
print("  14. 14_真实数据_Digits.png")
print("\n请查看这些图片以了解各算法的效果！")
print("=" * 80)


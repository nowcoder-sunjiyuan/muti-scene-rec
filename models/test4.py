import numpy as np
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use('TkAgg')

# 设置随机种子保证可重复性
np.random.seed(42)
emb_dim = 64  # 嵌入维度
total_points = 1024

# 构建四类嵌入空间----------------------------------------------------------
# 计算机类中心点（作为主要关联节点）
comp_center = np.random.randn(1, emb_dim) * 0.5

# 产品经理类：基于计算机类中心偏移（产生相关性）
pm_center = comp_center + np.random.randn(1, emb_dim)*0.2  # 与计算机类关联

# 机械类：基于计算机类中心不同方向偏移
mech_center = comp_center - np.random.randn(1, emb_dim)*0.3  # 与计算机类关联但远离PM

# 其他类：均匀分布的噪声点
other_center = np.zeros((1, emb_dim))  # 独立分布

# 生成各簇数据点----------------------------------------------------------
comp_embs = comp_center + np.random.randn(409, emb_dim)*0.25  # 最密集的簇
pm_embs = pm_center + np.random.randn(310, emb_dim)*0.35  # 中等密度
mech_embs = mech_center + np.random.randn(51, emb_dim)*0.25  # 最紧凑的簇
other_embs = other_center + np.random.randn(254, emb_dim)*1.2  # 分散分布

# 合并数据
embeddings = np.concatenate([comp_embs, pm_embs, mech_embs, other_embs])
labels = ['Computer']*409 + ['Product']*310 + ['Mechanical']*51 + ['Other']*254

# t-SNE降维--------------------------------------------------------------
tsne = TSNE(n_components=2, perplexity=35, random_state=42,
            early_exaggeration=16, learning_rate=150)
emb_2d = tsne.fit_transform(embeddings)

# 可视化设置--------------------------------------------------------------
plt.figure(figsize=(10, 8))
sns.set_style("whitegrid")

# 自定义调色板（CMYK配色方案）
palette = {
    'Computer': '#2F4FAA',    # 计算机蓝
    'Product': '#3CB371',     # 产品绿
    'Mechanical': '#FF6347',  # 机械红
    'Other': '#808080'        # 其他灰
}

# 绘制散点图（调整透明度防止重叠）
scatter = sns.scatterplot(
    x=emb_2d[:,0], y=emb_2d[:,1],
    hue=labels, palette=palette,
    alpha=0.65, s=45, edgecolor='w', linewidth=0.4,
    hue_order=['Computer','Product','Mechanical','Other']  # 控制图例顺序
)

# 添加图例和标签
plt.title("Item Domain Embedding Space", fontsize=14, pad=20)
plt.xlabel("t-SNE Dimension 1", fontsize=10)
plt.ylabel("t-SNE Dimension 2", fontsize=10)
plt.legend(
    title='Category',
    bbox_to_anchor=(1.25, 1),
    borderaxespad=0.,
    frameon=True,
    facecolor='#F5F5F5'
)

# 优化布局
plt.tight_layout()
plt.savefig('domain_clusters1.pdf', dpi=300, bbox_inches='tight')
plt.show()
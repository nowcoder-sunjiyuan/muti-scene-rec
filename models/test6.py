import numpy as np
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.use('TkAgg')

# 参数设置 --------------------------------------------------------------
np.random.seed(42)
emb_dim = 64
total_points = 1024

# 构建中心点体系 ----------------------------------------------------------
base_center = np.random.randn(1, emb_dim) * 0.5  # 计算机类中心

# 关联拓扑生成 -----------------------------------------------------------
comp_center = base_center  # 计算机
product_center = comp_center + np.random.randn(1, emb_dim) * 0.15  # 最强关联
operation_center = product_center + np.random.randn(1, emb_dim) * 0.20  # 次级关联
mechanical_center = comp_center - np.random.randn(1, emb_dim) * 0.15  # 原机械类中心
chip_center = mechanical_center + np.random.randn(1, emb_dim) * 0.2  # 芯片中心
other_center = np.random.randn(1, emb_dim) * 1.1  # 独立分布

# 生成各簇数据点 --------------------------------------------------------
clusters = [
    ('Computer', 409, comp_center, 0.4, 0.9),
    ('Product', 307, product_center, 0.4, 0.8),
    ('Operation', 102, operation_center, 0.50, 0.7),
    # 将机械类合并到Other，保留原始分布参数
    ('Other', 72, mechanical_center, 0.45, 0.6),
    ('Chip', 51, chip_center, 0.5, 0.5),
    ('Other', 31, other_center, 1.5, 0.1)
]

embeddings = []
labels = []
mechanical_mask = []  # 用于追踪原机械类数据
for name, n, center, noise, factor in clusters:
    cluster_embs = center + np.random.randn(n, emb_dim) * noise + comp_center * (1 - factor)
    embeddings.append(cluster_embs)

    # 特殊标记原机械类数据
    if name == 'Other' and n == 72:
        labels.extend(['Other'] * n)
        mechanical_mask.extend([True] * n)
    else:
        labels.extend([name] * n)
        mechanical_mask.extend([False] * n)
mechanical_mask = np.array(mechanical_mask)


# 添加跨类别关联 --------------------------------------------------------
def add_cross_relation(embs, src_class, tgt_class, strength=0.2):
    mask_src = np.array(labels) == src_class
    mask_tgt = np.array(labels) == tgt_class
    src_points = embs[mask_src]
    tgt_points = embs[mask_tgt]

    for i in np.random.choice(len(src_points), int(len(src_points) * strength)):
        nearest_idx = np.argmin(np.linalg.norm(src_points[i] - tgt_points, axis=1))
        direction = tgt_points[nearest_idx] - src_points[i]
        embs[mask_src][i] += direction * 0.15

    for j in np.random.choice(len(tgt_points), int(len(tgt_points) * strength)):
        nearest_idx = np.argmin(np.linalg.norm(tgt_points[j] - src_points, axis=1))
        direction = src_points[nearest_idx] - tgt_points[j]
        embs[mask_tgt][j] += direction * 0.15


# 更新关联规则（移除机械类相关项）
cross_relations = [
    ('Computer', 'Product', 0.25),
    ('Product', 'Operation', 0.55)
]

embeddings = np.concatenate(embeddings)
for src, tgt, strength in cross_relations:
    add_cross_relation(embeddings, src, tgt, strength)

# t-SNE降维 ------------------------------------------------------------
tsne = TSNE(
    n_components=2,
    perplexity=45,
    early_exaggeration=24,
    learning_rate=200,
    n_iter=1500,
    random_state=42
)
emb_2d = tsne.fit_transform(embeddings)

# 可视化配置 ------------------------------------------------------------
plt.figure(figsize=(12, 9))
sns.set_style("whitegrid", {'grid.linestyle': ':', 'grid.color': '#E0E0E0'})

# 调色板设置
palette = {
    'Computer': '#2E75B6',  # 科技蓝
    'Product': '#8FBC8F',  # 生态绿
    'Operation': '#FFA500',  # 运营橙
    'Chip': '#9370DB',  # 芯片紫
    'Other': '#A9A9A9'  # 中性灰
}

# 分层次绘制
layers = [
    ('Other', 0.4, 35),  # 普通Other数据
    ('Chip', 0.7, 50),
    ('Operation', 0.75, 55),
    ('Product', 0.8, 60),
    ('Computer', 0.85, 65)
]

# 先绘制普通Other数据
mask = (np.array(labels) == 'Other') & ~mechanical_mask
plt.scatter(
    emb_2d[mask, 0], emb_2d[mask, 1],
    color=palette['Other'],
    alpha=0.4,
    s=35,
    edgecolor='white',
    linewidth=0.3,
    zorder=0
)

# 单独绘制原机械类数据（保持原有分布特征）
plt.scatter(
    emb_2d[mechanical_mask, 0], emb_2d[mechanical_mask, 1],
    color='#A9A9A9',  # 更深的灰色
    alpha=0.7,
    s=45,
    edgecolor='white',
    linewidth=0.3,
    zorder=1,
    label='_nolegend_'  # 隐藏图例
)

# 绘制其他类别
for name, alpha, size in layers[1:]:
    mask = np.array(labels) == name
    plt.scatter(
        emb_2d[mask, 0], emb_2d[mask, 1],
        color=palette[name],
        alpha=alpha,
        s=size,
        edgecolor='white',
        linewidth=0.3,
        zorder=layers.index((name, alpha, size)) + 2
    )

# 图例配置（排除机械类）
legend_elements = [
                      plt.Line2D([0], [0], marker='o', color='w', label=name,
                                 markerfacecolor=color, markersize=10)
                      for name, color in palette.items() if name != 'Other'
                  ] + [
                      plt.Line2D([0], [0], marker='o', color='w', label='Other',
                                 markerfacecolor='#A9A9A9', markersize=10)
                  ]

plt.legend(
    handles=legend_elements,
    title='Category Cluster',
    bbox_to_anchor=(1.15, 1),
    borderpad=1.2,
    labelspacing=1.5,
    frameon=True,
    facecolor='#F8F8F8',
    edgecolor='#D0D0D0',
    title_fontsize=12
)

plt.title("Item Embedding t-SNE",
          fontsize=6, pad=10, color='#333333')
plt.xlabel("t-SNE-1", fontsize=10, labelpad=8)
plt.ylabel("t-SNE-2", fontsize=10, labelpad=8)

plt.savefig('merged_clusters.pdf', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
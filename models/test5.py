import numpy as np
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use('TkAgg')

# 参数设置 --------------------------------------------------------------
np.random.seed(42)
emb_dim = 64
total_points = 1024  # 保持总点数不变

# 构建中心点体系 ----------------------------------------------------------
base_center = np.random.randn(1, emb_dim) * 0.5  # 计算机类中心

# 关联拓扑生成（基于用户指定的关联顺序）
comp_center = base_center  # 计算机
product_center = comp_center + np.random.randn(1, emb_dim)*0.15  # 最强关联
operation_center = product_center + np.random.randn(1, emb_dim)*0.25  # 次级关联
mechanical_center = comp_center - np.random.randn(1, emb_dim)*0.15  # 中强关联
chip_center = mechanical_center + np.random.randn(1, emb_dim)*0.2  # 机械与芯片互相关联
other_center = np.random.randn(1, emb_dim)*1.1  # 独立分布

# 生成各簇数据点（调整密度和分散度）-----------------------------------------
clusters = [
    # 类别           样本数  中心点         噪声系数  关联强度因子
    ('Computer',    409,  comp_center,      0.4,    0.9),
    ('Product',     307,  product_center,   0.4,    0.8),
    ('Operation',   102,  operation_center, 0.50,   0.7),
    ('Mechanical',  72,   mechanical_center,0.35,   0.6),
    ('Chip',        51,   chip_center,      0.5,   0.5),
    ('Other',       31,   other_center,     1.5,    0.1)
]

embeddings = []
labels = []
for name, n, center, noise, factor in clusters:
    # 添加关联性噪声（基于计算机中心）
    related_noise = comp_center * (1 - factor)
    cluster_embs = center + np.random.randn(n, emb_dim)*noise + related_noise
    embeddings.append(cluster_embs)
    labels.extend([name]*n)


# 添加跨类别关联 ----------------------------------------------------------
def add_cross_relation(embs, src_class, tgt_class, strength=0.2):
    """ 创建类别间关联纽带 """
    mask_src = np.array(labels) == src_class
    mask_tgt = np.array(labels) == tgt_class
    src_points = embs[mask_src]
    tgt_points = embs[mask_tgt]

    # 建立双向引力（基于最近邻）
    for i in np.random.choice(len(src_points), int(len(src_points) * strength)):
        nearest_idx = np.argmin(np.linalg.norm(src_points[i] - tgt_points, axis=1))
        direction = tgt_points[nearest_idx] - src_points[i]
        embs[mask_src][i] += direction * 0.15  # 增强方向性

    for j in np.random.choice(len(tgt_points), int(len(tgt_points) * strength)):
        nearest_idx = np.argmin(np.linalg.norm(tgt_points[j] - src_points, axis=1))
        direction = src_points[nearest_idx] - tgt_points[j]
        embs[mask_tgt][j] += direction * 0.15


# 应用关联规则（基于用户定义的关联矩阵）
cross_relations = [
    ('Computer', 'Product', 0.25),
    ('Computer', 'Mechanical', 0.18),
    ('Product', 'Operation', 0.55),
    ('Mechanical', 'Chip', 0.3)
]

embeddings = np.concatenate(embeddings)
for src, tgt, strength in cross_relations:
    add_cross_relation(embeddings, src, tgt, strength)


# t-SNE参数优化 ----------------------------------------------------------
tsne = TSNE(
    n_components=2,
    perplexity=45,  # 增大以适应更多类别
    early_exaggeration=24,  # 增强初始分离度
    learning_rate=200,      # 加快收敛
    n_iter=1500,            # 增加迭代次数
    random_state=42
)
emb_2d = tsne.fit_transform(embeddings)

# 可视化配置 ------------------------------------------------------------
plt.figure(figsize=(12, 9))
sns.set_style("whitegrid", {'grid.linestyle': ':', 'grid.color': '#E0E0E0'})

# 优化配色方案（增强区分度）
palette = {
    'Computer': '#2E75B6',    # 科技蓝
    'Product': '#8FBC8F',     # 生态绿
    'Operation': '#FFA500',   # 运营橙
    'Mechanical': '#A9A9A9',  # 机械也是灰色
    'Chip': '#9370DB',        # 芯片紫
    'Other': '#A9A9A9'        # 中性灰
}

# 分层次绘制（提升可视化清晰度）
layers = [
    ('Other', 0.4, 35),       # 底层：其他类
    ('Mechanical', 0.6, 45),
    ('Chip', 0.7, 50),
    ('Operation', 0.75, 55),
    ('Product', 0.8, 60),
    ('Computer', 0.85, 65)    # 顶层：计算机类
]

for name, alpha, size in layers:
    mask = np.array(labels) == name
    plt.scatter(
        emb_2d[mask, 0], emb_2d[mask, 1],
        color=palette[name],
        alpha=alpha,
        s=size,
        edgecolor='white',
        linewidth=0.3,
        zorder=layers.index((name, alpha, size))  # 控制绘制顺序
    )

# 添加动态注释 ----------------------------------------------------------
from adjustText import adjust_text
texts = []
for class_name in ['Computer', 'Product', 'Chip']:  # 关键类别添加标签
    mask = np.array(labels) == class_name
    median = np.median(emb_2d[mask], axis=0)
    texts.append(plt.text(median[0], median[1], class_name,
             fontsize=10, weight='bold',
             color=palette[class_name], alpha=0.9))
adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', alpha=0.4))

# 高级图例配置 ----------------------------------------------------------
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label=name,
               markerfacecolor=color, markersize=10)
    for name, color in palette.items()
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

plt.title("Multi-Domain Embedding Space with Cross-Relations",
         fontsize=14, pad=18, color='#333333')
plt.xlabel("t-SNE 1", fontsize=10, labelpad=8)
plt.ylabel("t-SNE 2", fontsize=10, labelpad=8)

# 保存高清矢量图
plt.savefig('enhanced_domain_clusters.pdf',
           dpi=300,
           bbox_inches='tight',
           facecolor='white')
plt.show()
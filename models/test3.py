import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# 模拟数据：假设嵌入维度=64
np.random.seed(42)
user_emb = np.random.randn(1, 64) * 0.1  # 用户嵌入（中心点）
pos_embs = user_emb + np.random.randn(50, 64) * 0.3  # 正样本围绕用户
neg_embs = np.random.randn(200, 64) * 0.5 + 2.0       # 负样本分散

embeddings = np.concatenate([user_emb, pos_embs, neg_embs])
labels = ['User'] + ['Positive']*50 + ['Negative']*200

# t-SNE降维
tsne = TSNE(n_components=2, perplexity=25, random_state=42)
emb_2d = tsne.fit_transform(embeddings)

# 绘制
plt.figure(figsize=(8,6))
sns.scatterplot(x=emb_2d[:,0], y=emb_2d[:,1], hue=labels,
                palette={'User':'blue', 'Positive':'#2ca02c', 'Negative':'#d62728'},
                alpha=0.7, s=40, edgecolor='w', linewidth=0.3)
plt.xlabel("t-SNE Dimension 1", fontsize=12)
plt.ylabel("t-SNE Dimension 2", fontsize=12)
plt.title("item embedding space visualization", fontsize=14)
plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(alpha=0.2)
plt.savefig('tsne_simulated.pdf', bbox_inches='tight', dpi=300)
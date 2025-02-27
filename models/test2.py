import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorboard
from tensorboard.plugins import projector

# 参数设置
num_samples = 100  # 样本数量
num_features = 2  # 特征数量
num_clusters = 3  # 簇的数量

# 生成随机数据
np.random.seed(42)
data = np.random.rand(num_samples, num_features)  # 随机数据点
labels = np.random.randint(num_clusters, size=num_samples)  # 随机标签

# 添加一些噪声
data += np.random.normal(0, 0.1, data.shape)

# 创建元数据文件
metadata_file = "metadata.tsv"
with open(metadata_file, "w") as f:
    f.write("Index\tLabel\tColor\n")
    for i in range(num_samples):
        label = labels[i]
        color = ["red", "green", "blue"][label]  # 根据标签分配颜色
        f.write(f"{i}\t{label}\t{color}\n")

# 创建嵌入向量
embedding = tf.Variable(data, name="embedding")

# 创建日志目录
log_dir = "E://logs/embedding"

# 创建 summary writer
summary_writer = tf.summary.create_file_writer(log_dir)

# 将嵌入向量写入日志
with summary_writer.as_default():
    projector_config = projector.ProjectorConfig()
    embedding_config = projector_config.embeddings.add()
    embedding_config.tensor_name = embedding.name
    embedding_config.metadata_path = metadata_file

# 使用日志目录路径而不是 summary writer 对象
projector.visualize_embeddings(log_dir, projector_config)

print(f"嵌入向量已保存到 {log_dir}")
print(f"元数据文件已保存到 {metadata_file}")

# 启动 TensorBoard
# 运行以下命令以启动 TensorBoard:
# tensorboard --logdir=logs
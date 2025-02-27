import os
import numpy as np
import tensorflow as tf
from tensorboard.plugins.projector import ProjectorConfig

# 生成模拟数据
num_users = 2000
embeddings = np.random.randn(num_users, 64).astype(np.float32)
scenarios = ["推荐", "职场", "知识", "最新"]
scenario_labels = np.random.choice(scenarios, num_users)

# 定义日志目录
log_dir = os.path.join("logs", "emb_projector")
os.makedirs(log_dir, exist_ok=True)

# 保存元数据文件 (UTF-8编码)
metadata_path = os.path.join(log_dir, "metadata.tsv")
with open(metadata_path, "w", encoding="utf-8") as f:
    f.write("UserID\tScenario\n")
    for i, label in enumerate(scenario_labels):
        f.write(f"user_{i}\t{label}\n")

# 创建并保存嵌入变量
embeddings_var = tf.Variable(embeddings, name="user_embeddings")
checkpoint = tf.train.Checkpoint(embedding=embeddings_var)
checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

# 生成投影仪配置文件
config = ProjectorConfig()
embedding_config = config.embeddings.add()
embedding_config.tensor_name = "user_embeddings/.ATTRIBUTES/VARIABLE_VALUE"
embedding_config.metadata_path = "metadata.tsv"  # 相对于日志目录的路径

config_path = os.path.join(log_dir, "projector_config.pbtxt")
with open(config_path, "w", encoding="utf-8") as f:
    f.write(str(config))

# 生成空事件文件触发TensorBoard识别
with tf.summary.create_file_writer(log_dir).as_default():
    tf.summary.scalar("dummy", 0.0, step=0)
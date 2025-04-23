import tensorflow as tf


def batch_contrastive_loss(user_vec, item_vec, temperature=0.1):
    """
    用户-物品对比学习损失
    Args:
        user_vec: shape (batch_size, dim) 比如(1024, 64)
        item_vec: shape (batch_size, dim) 比如(1024, 64)
        temperature: 温度系数
    """
    # 计算相似度矩阵 (1024, 1024)
    sim_matrix = tf.matmul(user_vec, item_vec, transpose_b=True)
    sim_matrix = sim_matrix / temperature

    # 构造标签矩阵 - 对角线为正样本
    batch_size = tf.shape(user_vec)[0]
    labels = tf.eye(batch_size)

    # 计算损失
    log_softmax = tf.nn.log_softmax(sim_matrix, axis=1)
    loss = -tf.reduce_sum(labels * log_softmax) / tf.cast(batch_size, tf.float32)

    return loss


def bidirectional_contrastive_loss(user_vec, item_vec, temperature=0.1):
    """
    双向对比学习损失
    """
    loss_user = batch_contrastive_loss(user_vec, item_vec, temperature)
    loss_item = batch_contrastive_loss(item_vec, user_vec, temperature)
    return (loss_user + loss_item) / 2.0


# 使用示例
@tf.function
def train_step(user_vec, item_vec, optimizer):
    with tf.GradientTape() as tape:
        loss = bidirectional_contrastive_loss(user_vec, item_vec)

    # 计算梯度并更新
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


# Hyperparameters
batch_size = 1024
embedding_dim = 64
temperature = 0.1  # Temperature parameter for InfoNCE

# User and item embeddings as inputs
user_embeddings = tf.placeholder(tf.float32, shape=(batch_size, embedding_dim))
item_embeddings = tf.placeholder(tf.float32, shape=(batch_size, embedding_dim))

# Calculate similarity matrix
scores = tf.matmul(user_embeddings, item_embeddings, transpose_b=True)  # Shape: (batch_size, batch_size)


# InfoNCE loss calculation
def info_nce_loss(scores, temperature):
    # Create labels for positive samples. These are just the diagonal indices.
    labels = tf.range(batch_size)

    # Scale logits by temperature and compute loss
    logits = scores / temperature

    # Compute cross-entropy loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)

    return tf.reduce_mean(loss)


# Calculate contrastive loss
contrastive_loss = info_nce_loss(scores, temperature)

# Define optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(contrastive_loss)

# TensorFlow session to execute computation
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Dummy data for demonstration
    user_data = np.random.rand(batch_size, embedding_dim)
    item_data = np.random.rand(batch_size, embedding_dim)

    # Run optimization
    _, loss_value = sess.run([optimizer, contrastive_loss],
                             feed_dict={user_embeddings: user_data, item_embeddings: item_data})

    print(f"Contrastive Loss: {loss_value}")
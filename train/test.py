import tensorflow as tf

def compute_loss(self, scene_user_vecs, scene_item_vecs, scene_masks):
    """
    多场景对比学习损失计算
    Args:
        scene_user_vecs: 列表,包含4个场景的用户向量 [(200,64), (345,64), (400,64), (65,64)]
        scene_item_vecs: 列表,包含4个场景的物品向量 [(200,64), (345,64), (400,64), (65,64)]
        scene_masks: 列表,包含每个场景的交互mask
    """
    total_loss = 0.0
    num_scenes = len(scene_user_vecs)

    for scene_idx in range(num_scenes):
        current_users = scene_user_vecs[scene_idx]  # (scene_size, 64)
        current_items = scene_item_vecs[scene_idx]  # (scene_size, 64)
        current_mask = scene_masks[scene_idx]  # (scene_size, scene_size) 交互矩阵
        scene_size = tf.shape(current_users)[0]

        # 计算场景内所有用户-物品对的相似度
        sim_matrix = tf.matmul(current_users, current_items, transpose_b=True)  # (scene_size, scene_size)
        sim_matrix = sim_matrix / self.temperature

        # 使用交互矩阵作为正样本标签
        labels = current_mask

        # 计算用户视角的损失
        user_loss = self.compute_single_loss(sim_matrix, labels)

        # 计算物品视角的损失
        item_loss = self.compute_single_loss(tf.transpose(sim_matrix), tf.transpose(labels))

        scene_loss = (user_loss + item_loss) / 2.0
        total_loss += scene_loss

    return total_loss / num_scenes


def compute_single_loss(self, sim_matrix, labels):
    """
    计算单个视角的对比损失
    Args:
        sim_matrix: 相似度矩阵 (batch_size, batch_size)
        labels: 交互标签矩阵 (batch_size, batch_size)
    """
    exp_sim = tf.exp(sim_matrix)

    # 计算正样本的损失
    pos_mask = labels
    pos_sim = tf.reduce_sum(exp_sim * pos_mask, axis=1)  # 仅考虑正样本对

    # 计算负样本的损失
    neg_mask = 1 - labels  # 非交互样本为负样本
    neg_sim = tf.reduce_sum(exp_sim * neg_mask, axis=1)

    loss = -tf.math.log(pos_sim / (pos_sim + neg_sim + 1e-8))
    return tf.reduce_mean(loss)
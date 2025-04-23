import tensorflow as tf

def compute_loss(self, scene_user_vecs, scene_item_vecs, scene_masks):
    """
    多场景对比学习损失计算
    Args:
        scene_user_vecs: 列表，每个场景的用户向量 [(200,64), (345,64), (400,64), (65,64)]
        scene_item_vecs: 列表，每个场景的物品向量 
        scene_masks: 列表，每个场景的交互mask
    """
    num_scenes = len(scene_user_vecs)
    total_loss = 0.0

    # 1. 构建全场景负样本
    all_items = tf.concat(scene_item_vecs, axis=0)  # 所有场景的物品向量
    all_users = tf.concat(scene_user_vecs, axis=0)  # 所有场景的用户向量

    for scene_idx in range(num_scenes):
        current_users = scene_user_vecs[scene_idx]  # (scene_size, 64)
        current_items = scene_item_vecs[scene_idx]  # (scene_size, 64)
        scene_size = tf.shape(current_users)[0]

        # 2. 计算当前场景的正样本相似度
        pos_sim = tf.matmul(current_users, current_items, transpose_b=True)  # (scene_size, scene_size)
        pos_sim = pos_sim / self.temperature

        # 3. 计算与全场景的负样本相似度
        neg_sim_users = tf.matmul(current_users, all_items, transpose_b=True)  # (scene_size, total_items)
        neg_sim_users = neg_sim_users / self.temperature

        # 4. 用户视角：当前场景正样本 vs 全场景负样本
        user_loss = 0.0
        for i in range(scene_size):
            positive = pos_sim[i, i]  # 当前用户的正样本相似度
            negative = neg_sim_users[i]  # 当前用户与所有物品的相似度

            # 计算InfoNCE loss
            numerator = tf.exp(positive)
            denominator = numerator + tf.reduce_sum(tf.exp(negative))
            user_loss += -tf.math.log(numerator / denominator)

        user_loss = user_loss / scene_size

        # 5. 物品视角的计算（类似上面的步骤）
        neg_sim_items = tf.matmul(current_items, all_users, transpose_b=True)
        neg_sim_items = neg_sim_items / self.temperature

        item_loss = 0.0
        for i in range(scene_size):
            positive = pos_sim[i, i]
            negative = neg_sim_items[i]

            numerator = tf.exp(positive)
            denominator = numerator + tf.reduce_sum(tf.exp(negative))
            item_loss += -tf.math.log(numerator / denominator)

        item_loss = item_loss / scene_size

        # 6. 当前场景的总损失
        scene_loss = (user_loss + item_loss) / 2.0
        total_loss += scene_loss

    # 7. 所有场景的平均损失
    return total_loss / num_scenes
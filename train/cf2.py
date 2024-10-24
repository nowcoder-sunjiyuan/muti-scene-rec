import tensorflow as tf
from data_process import dataset_process
from models.FeatureEmbModel import FeatureEmbModel
import utils.nn_utils as nn
import keras
import time
from mmoe import mmoe


def cross_entropy(target, pos, neg):
    # 计算正负样本的logits
    pos_logits = tf.reduce_sum(pos * target, axis=-1)
    neg_logits = tf.reduce_sum(neg * target, axis=-1)

    # 计算损失
    # pos_loss = -tf.math.log(tf.sigmoid(pos_logits) + 1e-24)  # 距离更近
    # neg_loss = -tf.math.log(1 - tf.sigmoid(neg_logits) + 1e-24)  # 距离更远
    pos_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(pos_logits), logits=pos_logits)
    neg_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(neg_logits), logits=neg_logits)

    # 由于不需要掩码，我们直接计算平均损失
    loss = tf.reduce_mean(pos_loss + neg_loss)

    return loss


train_file, valid_file = dataset_process.win_train_test_file()
data_process = dataset_process.DatasetProcess()
train_dataset = data_process.create_dataset_cl(train_file, 1024)
# rec_cf_data_iter = enumerate(train_dataset)
# for (i, (input_dict)) in rec_cf_data_iter:
#     print(input_dict)

valid_dataset = data_process.create_dataset_cl(valid_file, 1024)

optimizer = keras.optimizers.Adam(0.0003)
# emb层
feature_emb_model = FeatureEmbModel()
# 全连接层
mmoe_layer = mmoe.MMoE(units=128, num_experts=8, num_tasks=1)
ctr_output = nn.FullyConnectedTower([64, 32, 1], 'ctr', 'relu', 'sigmoid')

# ctr_output = keras.layers.Dense(1, activation='sigmoid')
num_batches = 0
# 开始训练
for epoch in range(3):
    total_loss = 0.0
    total_ctr_loss = 0.0
    train_auc_metric = keras.metrics.AUC(from_logits=False)
    rec_cf_data_iter = enumerate(train_dataset)
    last_time = None  # 记录上一次结束时间
    for (i, (input_dict, target_pos, target_neg, label)) in rec_cf_data_iter:  # 某个批次

        start_time = time.time()
        with tf.GradientTape() as tape:
            start_time_forward_pass = time.time()
            target_emb = feature_emb_model.call(input_dict, mode="embedding")  # (1024, 304)
            target_pos_emb = feature_emb_model.call(target_pos, mode="embedding")  # (1024, 304)
            target_neg_emb = feature_emb_model.call(target_neg, mode="embedding")  # (1024, 304)

            end_time_forward_pass = time.time()
            start_time_loss_calculation = time.time()
            # 对比学习损失
            rec_cf_loss = cross_entropy(target_emb, target_pos_emb, target_neg_emb)
            # 交叉熵
            ctr_predictions = ctr_output(mmoe_layer(target_emb)[0])
            ctr_loss = keras.losses.binary_crossentropy(label['label'], ctr_predictions)
            ctr_loss = tf.reduce_mean(ctr_loss)

            combined_loss = 0.2 * rec_cf_loss + 0.8 * ctr_loss
            end_time_loss_calculation = time.time()

            train_auc_metric.update_state(label['label'], ctr_predictions)

        start_time_backward_pass = time.time()
        gradients = tape.gradient(combined_loss, feature_emb_model.trainable_variables + ctr_output.trainable_variables + mmoe_layer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, feature_emb_model.trainable_variables + ctr_output.trainable_variables + mmoe_layer.trainable_variables))
        end_time_backward_pass = time.time()
        last_time = end_time = time.time()

        num_batches += 1
        total_loss += combined_loss.numpy()
        total_ctr_loss += ctr_loss

        print(
            f"\rEpoch {epoch}, Step {i}, avg total loss: {total_loss / num_batches}, avg CTR loss: {total_ctr_loss / num_batches}, Train AUC: {train_auc_metric.result().numpy()}",
            end='', flush=True)
        if i % 4000 == 0:
            print(
                f"Epoch {epoch}, Step {i}, avg total loss: {total_loss / num_batches}, avg CTR loss: {total_ctr_loss / num_batches}, Train AUC: {train_auc_metric.result().numpy()}",
                end='', flush=True)
        # print(
        #     f"Step {i} time stats - Forward Pass: {end_time_forward_pass - start_time_forward_pass}s, "
        #     f"Loss Calculation: {end_time_loss_calculation - start_time_loss_calculation}s, "
        #     f"Backward Pass: {end_time_backward_pass - start_time_backward_pass}s")

    # Epoch结束，打印平均损失
    print(f"Epoch {epoch}, avg total loss: {total_loss / num_batches}, avg CTR loss: {total_ctr_loss/num_batches}, Train AUC: {train_auc_metric.result().numpy()}", end='', flush=True)

    # 创建AUC指标实例
    auc_metric = keras.metrics.AUC(from_logits=False)

    # 验证集评估
    auc_metric.reset_state()  # 重置状态，以便于新的评估开始
    for input_dict, target_pos, target_neg, label in valid_dataset:
        target_emb = feature_emb_model.call(input_dict, mode="embedding")
        ctr_predictions = ctr_output(mmoe_layer(target_emb))
        auc_metric.update_state(label['label'], ctr_predictions)  # 更新状态，累积预测和标签

    auc_score = auc_metric.result().numpy()  # 计算AUC值
    print(f"Validation AUC: {auc_score}")

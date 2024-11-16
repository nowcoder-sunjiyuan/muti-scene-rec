import tensorflow as tf
from data_process import dataset_process
from models.FeatureEmbModel import FeatureEmbModel
from models.Attention import SelfAttention
import utils.nn_utils as nn
import keras
import time
from mmoe import mmoe
import trainers


train_file, valid_file = dataset_process.win_train_test_file()
data_process = dataset_process.DatasetProcess()
train_dataset = data_process.create_dataset_cl(train_file, 1024)
valid_dataset = data_process.create_dataset_cl(valid_file, 1024)

optimizer = keras.optimizers.Adam(0.0003)
mc = trainers.MetricsControllerOne()
# emb层
feature_emb_model = FeatureEmbModel()
# 注意力层
attention_model = SelfAttention(32)
# mmoe
mmoe_layer = mmoe.MMoE(units=256, num_experts=4, num_tasks=1)
# 对比学习表达层
cl_expression_layer = nn.FullyConnectedTower([256], 'cl_expression', 'relu', 'relu')
# ctr预估层
ctr_output = nn.FullyConnectedTower([128, 64, 1], 'ctr', 'relu', 'sigmoid')
num_batches = 0
# 开始训练
for epoch in range(3):
    total_loss = 0.0
    total_ctr_loss = 0.0
    rec_cf_data_iter = enumerate(train_dataset)
    last_time = None  # 记录上一次结束时间

    mc.reset_train_metrics()

    for (i, (input_dict, target_pos, target_neg, seq_dict, label)) in rec_cf_data_iter:  # 某个批次
        with tf.GradientTape() as tape:
            # 这里先取出平台
            platform = input_dict['platform']

            target_emb = feature_emb_model.call(input_dict, mode="embedding")  # (1024, 304)
            target_pos_emb = feature_emb_model.call(target_pos, mode="embedding")  # (1024, 304)
            target_neg_emb = feature_emb_model.call(target_neg, mode="embedding")  # (1024, 304)
            seq_dict = feature_emb_model.call(seq_dict, mode="embedding_seq")

            # 序列处理
            attention_result = attention_model(seq_dict['hist_entity_id'])  # (1024, 20, 32)
            reduce_mean_attention_emb = tf.reduce_mean(attention_result, axis=1)
            # input_emb = tf.concat(target_emb, reduce_mean_attention_emb, axis=1)

            # 对比学习表示层
            cl_target = cl_expression_layer(target_emb)  # (1024, 256)
            cl_pos = cl_expression_layer(target_pos_emb)  # (1024, 256)
            cl_neg = cl_expression_layer(target_neg_emb)  # (1024, 256)

            # 对比学习损失
            cl_loss = nn.cross_entropy(cl_target, cl_pos, cl_neg)

            # 对比学习表达和序列特征拼接
            input_emb = tf.concat([cl_target, reduce_mean_attention_emb], axis=-1)

            # 交叉熵
            mmoe_emb = mmoe_layer(input_emb)  # (1024, 256)
            ctr_predictions = ctr_output(mmoe_emb[0])
            ctr_loss = keras.losses.binary_crossentropy(label['label'], ctr_predictions)
            ctr_loss = tf.reduce_mean(ctr_loss)
            combined_loss = cl_loss + ctr_loss

            # 更新指标
            platforms = tf.squeeze(platform)
            mc.update_train_metrics(label['label'], ctr_predictions, platforms)

        trainable_variables = feature_emb_model.trainable_variables + mmoe_layer.trainable_variables + cl_expression_layer.trainable_variables + ctr_output.trainable_variables

        gradients = tape.gradient(combined_loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))

        num_batches += 1
        total_loss += combined_loss.numpy()

        # 打印指标信息
        mc.log_train_metrics(True, epoch, i, total_loss / num_batches)

    # Epoch结束，打印指标信息
    mc.log_train_metrics(False, epoch, 0, total_loss / num_batches)

    print("\n")
    # 重置验证集指标
    mc.reset_valid_metrics()
    for input_dict, target_pos, target_neg, seq_dict, label in valid_dataset:

        # 序列层
        seq_dict = feature_emb_model.call(seq_dict, mode="embedding_seq")
        attention_result = attention_model(seq_dict['hist_entity_id'])  # (1024, 20, 32)
        reduce_mean_attention_emb = tf.reduce_mean(attention_result, axis=1)

        # 对比学习层
        target_emb = feature_emb_model.call(input_dict, mode="embedding")
        cl_target = cl_expression_layer(target_emb)

        # 拼接
        input_emb = tf.concat([cl_target, reduce_mean_attention_emb], axis=-1)

        # mmoe层
        mmoe_emb = mmoe_layer(cl_target)

        # 预测
        ctr_predictions = ctr_output(mmoe_emb[0])
        # 更新验证集指标
        platforms = tf.squeeze(input_dict['platform'])
        mc.update_valid_metrics(label['label'], ctr_predictions, platforms)
        mc.log_valid_metrics()

    # 打印验证集指标
    mc.log_valid_metrics()
    print("\n")
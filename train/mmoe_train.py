import tensorflow as tf
from data_process import dataset_process
from models.FeatureEmbModel import FeatureEmbModel
import utils.nn_utils as nn
import keras
import time
from mmoe import mmoe
import trainers

train_file, valid_file = dataset_process.win_train_test_file()
data_process = dataset_process.DatasetProcess()
train_dataset = data_process.create_dataset_normal(train_file, 1024)
valid_dataset = data_process.create_dataset_normal(valid_file, 1024)

optimizer = keras.optimizers.Adam(0.0003)
mc = trainers.MetricsControllerOne()
# emb层
feature_emb_model = FeatureEmbModel()
# mmoe
mmoe_layer = mmoe.MMoE(units=256, num_experts=4, num_tasks=1)
# ctr预估
ctr_output = nn.FullyConnectedTower([128, 64, 1], 'ctr', 'relu', 'sigmoid')


# 开始训练
for epoch in range(3):
    total_loss = 0.0
    num_batches = 0
    rec_cf_data_iter = enumerate(train_dataset)
    last_time = None  # 记录上一次结束时间

    mc.reset_train_metrics()

    for (i, (input_dict, label)) in rec_cf_data_iter:  # 某个批次
        with tf.GradientTape() as tape:
            target_emb = feature_emb_model.call(input_dict, mode="embedding")  # (1024, 304)
            mmoe_emb = mmoe_layer(target_emb)
            ctr_predictions = ctr_output(mmoe_emb[0])

            # 交叉熵
            ctr_loss = keras.losses.binary_crossentropy(label['label'], ctr_predictions)
            ctr_loss = tf.reduce_mean(ctr_loss)

            # 更新指标
            platforms = tf.squeeze(input_dict['platform'])
            mc.update_train_metrics(label['label'], ctr_predictions, platforms)

        trainable_variables = feature_emb_model.trainable_variables + mmoe_layer.trainable_variables + ctr_output.trainable_variables
        gradients = tape.gradient(ctr_loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))

        num_batches += 1
        total_loss += ctr_loss

        # 打印指标信息
        mc.log_train_metrics(True, epoch, i, total_loss/num_batches)

    # Epoch结束，打印指标信息
    mc.log_train_metrics(False, epoch, 0, total_loss/num_batches)
    print("\n")

    # 重置验证集指标
    mc.reset_valid_metrics()
    for input_dict, label in valid_dataset:
        target_emb = feature_emb_model.call(input_dict, mode="embedding")
        mmoe_emb = mmoe_layer(target_emb)
        ctr_predictions = ctr_output(mmoe_emb[0])
        # 更新验证集指标
        platforms = tf.squeeze(input_dict['platform'])
        mc.update_valid_metrics(label['label'], ctr_predictions, platforms)
        mc.log_valid_metrics()
    # 打印验证集指标
    mc.log_valid_metrics()
    print("\n")
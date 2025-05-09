import tensorflow as tf
from data_process import dataset_process
from models.FeatureEmbModel import FeatureEmbModel
from models.Attention import SelfAttention, EncoderLayerSelfAtt, EncoderLayerMultiHeadAtt, EncoderLayerTimeEn, \
    EncoderLayerConvTimeEn
import utils.nn_utils as nn
import keras
import time
from mmoe import mmoe
import trainers
import datetime


# 输入的参数
encoder_type = "v2"
cl_type = "v1"

train_file, valid_file = dataset_process.win_train_test_file()
data_process = dataset_process.DatasetProcess()
train_dataset = data_process.create_dataset_cl(train_file, 1024)
valid_dataset = data_process.create_dataset_cl(valid_file, 1024)

optimizer = keras.optimizers.Adam(0.0003)
mc = trainers.MetricsControllerOne()
# emb层
feature_emb_model = data_process.feature_parse_model
# encode层, 这里的序列长度是20
if encoder_type == "v1":
    attention_model = EncoderLayerSelfAtt(32, 32)
elif encoder_type == "v2":
    attention_model = EncoderLayerMultiHeadAtt(32, 32)
elif encoder_type == "v3":
    attention_model = EncoderLayerTimeEn(32, 32, 20)
elif encoder_type == "v4":
    attention_model = EncoderLayerConvTimeEn(32, 32, 20)
else:
    attention_model = EncoderLayerSelfAtt(32, 32)
# mmoe
mmoe_layer = mmoe.MMoE(units=256, num_experts=4, num_tasks=1)
# 对比学习表达层
cl_expression_layer = nn.FullyConnectedTower([256], 'cl_expression', 'relu', 'relu')
# ctr预估层
ctr_output = nn.FullyConnectedTower([128, 64, 1], 'ctr', 'relu', 'sigmoid')
num_batches = 0


def print_trainable_params(models_list):
    total_params = 0
    for i, model in enumerate(models_list):
        params = model.count_params()
        total_params += params
        print(f"Model Component {i+1} ({model.name}) Params: {params:,}")
    print(f"\nTotal Trainable Parameters: {total_params:,}\n")

# 取一个虚拟样本触发模型构建
dummy_sample = next(iter(train_dataset))
dummy_input, dummy_target_pos, dummy_target_neg, dummy_seq, dummy_label = dummy_sample

# 触发 feature_emb_model 构建
_ = feature_emb_model.call(dummy_input, mode="embedding")
_ = feature_emb_model.call(dummy_seq, mode="embedding_seq")

feature_emb_model.check_built_status()
# 触发 attention_model 构建
dummy_seq_emb = feature_emb_model.call(dummy_seq, mode="embedding_seq")
_ = attention_model(dummy_seq_emb['hist_entity_id'],  training=True)

# 触发后续层构建（确保计算图完整）
dummy_target_emb = feature_emb_model.call(dummy_input, mode="embedding")
dummy_cl = cl_expression_layer(dummy_target_emb)
dummy_attn = attention_model(dummy_seq_emb['hist_entity_id'], training=True)
dummy_attn_mean = tf.reduce_mean(dummy_attn, axis=1)
dummy_mmoe_input = tf.concat([dummy_cl, dummy_attn_mean], axis=-1)
_ = mmoe_layer(dummy_mmoe_input)
_ = ctr_output(mmoe_layer(dummy_mmoe_input)[0])

# 此时再打印参数即可正确显示
# print_trainable_params([...])
print_trainable_params([
    feature_emb_model,
    attention_model,
    mmoe_layer,
    cl_expression_layer,
    ctr_output
])


# 定义 TensorBoard 回调
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = "E://logs/" + current_time
train_log_dir = log_dir + '/train'
val_log_dir = log_dir + '/val'

train_summary_writer = tf.summary.create_file_writer(train_log_dir)

# 开始训练
for epoch in range(3):
    total_loss = 0.0
    total_ctr_loss = 0.0
    rec_cf_data_iter = enumerate(train_dataset)
    last_time = None  # 记录上一次结束时间

    mc.reset_train_metrics()

    for (i, (input_dict, target_pos, target_neg, seq_dict, label)) in rec_cf_data_iter:  # 某个批次
    # for sample in train_dataset.take(1):
    #     input_dict, target_pos, target_neg, seq_dict, label = sample
        with tf.GradientTape() as tape:
            # 这里先取出平台
            platform = input_dict['platform']

            target_emb = feature_emb_model.call(input_dict, mode="embedding")  # (1024, 304)
            target_pos_emb = feature_emb_model.call(target_pos, mode="embedding")  # (1024, 304)
            target_neg_emb = feature_emb_model.call(target_neg, mode="embedding")  # (1024, 304)
            seq_dict = feature_emb_model.call(seq_dict, mode="embedding_seq")

            # 序列处理
            attention_result = attention_model(seq_dict['hist_entity_id'], training=True)  # (1024, 20, 32)
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
            if cl_type == "v1":
                combined_loss = ctr_loss
            else:
                combined_loss = cl_loss + ctr_loss

            # 更新指标
            platforms = tf.squeeze(platform)

            att_weight = attention_model.attention_weights()

            with train_summary_writer.as_default():
                step = optimizer.iterations.numpy()
                tf.summary.scalar('combined_loss', combined_loss, step=optimizer.iterations)
                tf.summary.scalar('ctr_loss', ctr_loss, step=optimizer.iterations)
                tf.summary.scalar('cl_loss', cl_loss, step=optimizer.iterations)
                # 可视化第一个样本、第一个头的注意力热力图
                attention_head = att_weight[0, 0]  # 取第一个样本、第一个头
                attention_head = (attention_head - tf.reduce_min(attention_head)) / \
                                 (tf.reduce_max(attention_head) - tf.reduce_min(attention_head) + 1e-8)
                attention_img = tf.expand_dims(attention_head, axis=0)  # 添加batch维度
                attention_img = tf.expand_dims(attention_img, axis=-1)  # 添加通道维度
                attention_img = tf.repeat(attention_img, 3, axis=-1)  # 转换为RGB格式

                tf.summary.image("Attention Heatmap", attention_img, step=step, max_outputs=1)


            mc.update_train_metrics(label['label'], ctr_predictions, platforms)

        if cl_type == "v1":
            trainable_variables = feature_emb_model.trainable_variables + mmoe_layer.trainable_variables + ctr_output.trainable_variables
        else:
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
        attention_result = attention_model(seq_dict['hist_entity_id'], training=False)  # (1024, 20, 32)
        reduce_mean_attention_emb = tf.reduce_mean(attention_result, axis=1)

        # 对比学习层
        target_emb = feature_emb_model.call(input_dict, mode="embedding")
        cl_target = cl_expression_layer(target_emb)

        # 拼接
        input_emb = tf.concat([cl_target, reduce_mean_attention_emb], axis=-1)

        # mmoe层
        mmoe_emb = mmoe_layer(input_emb)

        # 预测
        ctr_predictions = ctr_output(mmoe_emb[0])
        # 更新验证集指标
        platforms = tf.squeeze(input_dict['platform'])
        mc.update_valid_metrics(label['label'], ctr_predictions, platforms)
        mc.log_valid_metrics()

    # 打印验证集指标
    mc.log_valid_metrics()
    print("\n")

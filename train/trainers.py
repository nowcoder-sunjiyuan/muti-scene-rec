import keras
import tensorflow as tf

platform_desc = {
    2: "IOS",
    3: "Android",
    4: "web"
}


class MetricsControllerOne:

    def __init__(self):
        self.train_auc_metric = keras.metrics.AUC(from_logits=False)
        self.valid_auc_metric = keras.metrics.AUC(from_logits=False)
        self.train_platform_auc_metrics = {
            2: keras.metrics.AUC(from_logits=False),
            3: keras.metrics.AUC(from_logits=False),
            4: keras.metrics.AUC(from_logits=False)
        }
        self.valid_platform_auc_metrics = {
            2: keras.metrics.AUC(from_logits=False),
            3: keras.metrics.AUC(from_logits=False),
            4: keras.metrics.AUC(from_logits=False)
        }

    def reset_train_metrics(self):
        self.train_auc_metric.reset_state()
        for metric in self.train_platform_auc_metrics.values():
            metric.reset_state()

    def reset_valid_metrics(self):
        self.valid_auc_metric.reset_state()
        for metric in self.valid_platform_auc_metrics.values():
            metric.reset_state()

    def update_train_metrics(self, label, prediction, platforms):
        self.train_auc_metric.update_state(label, prediction)

        # 获取平台的唯一值
        unique_platforms = tf.constant([2, 3, 4], dtype=tf.int64)  # (3,)

        # 遍历每个唯一平台
        for platform in unique_platforms:
            # 找到属于当前平台的样本索引
            platform_indices = tf.where(platforms == platform)  # platforms: (1024, ) result：(indices,1)
            platform_indices = tf.squeeze(platform_indices)  # (indices,)

            # 根据索引提取相应的标签和预测
            platform_labels = tf.gather(label, platform_indices)
            platform_predictions = tf.gather(prediction, platform_indices)

            # 更新该平台的指标
            if platform.numpy() in self.train_platform_auc_metrics:
                self.train_platform_auc_metrics[platform.numpy()].update_state(platform_labels, platform_predictions)

    def update_valid_metrics(self, label, prediction, platforms):
        self.valid_auc_metric.update_state(label, prediction)

        # 获取平台的唯一值
        unique_platforms = tf.constant([2, 3, 4], dtype=tf.int64)  # (3,)

        # 遍历每个唯一平台
        for platform in unique_platforms:
            # 找到属于当前平台的样本索引
            platform_indices = tf.where(platforms == platform)  # platforms: (1024, ) result：(indices,1)
            platform_indices = tf.squeeze(platform_indices)  # (indices,)

            # 根据索引提取相应的标签和预测
            platform_labels = tf.gather(label, platform_indices)
            platform_predictions = tf.gather(prediction, platform_indices)

            # 更新该平台的指标
            if platform.numpy() in self.valid_platform_auc_metrics:
                self.valid_platform_auc_metrics[platform.numpy()].update_state(platform_labels, platform_predictions)

    def log_train_metrics(self, in_epoch, epoch, step, avg_ctr_loss):
        if in_epoch:
            platform_output_log = ''
            for platform in self.train_platform_auc_metrics.keys():
                platform_output_log += f"{platform_desc[platform]} train AUC: {self.train_platform_auc_metrics[platform].result().numpy()} ,"
            print(
                f"\rEpoch {epoch},Step {step},avg CTR loss: {avg_ctr_loss},Train AUC: {self.train_auc_metric.result().numpy()},{platform_output_log}",
                end='', flush=True)
        else:
            platform_output_log = ''
            for platform in self.train_platform_auc_metrics.keys():
                platform_output_log += f"{platform_desc[platform]} train AUC: {self.train_platform_auc_metrics[platform].result().numpy()} ,"
            print(
                f"\rEpoch {epoch},avg CTR loss: {avg_ctr_loss},Train AUC: {self.train_auc_metric.result().numpy()},{platform_output_log}",
                end='', flush=True)

    def log_valid_metrics(self):
        platform_output_log = ''
        for platform in self.valid_platform_auc_metrics.keys():
            platform_output_log += f"{platform_desc[platform]} AUC: {self.valid_platform_auc_metrics[platform].result().numpy()} ,"
        print(f"\rValidation AUC: {self.valid_auc_metric.result().numpy()}, {platform_output_log}", end='', flush=True)

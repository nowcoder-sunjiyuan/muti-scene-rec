import random
import copy
import tensorflow as tf


class Random(object):
    """Randomly pick one data augmentation type every time call"""

    def __init__(self, gamma=0.1, features_name=None):
        if features_name is None:
            features_name = []
        self.data_augmentation_methods = [Mask(gamma=gamma, features_name=features_name)]
        print("Total augmentation numbers: ", len(self.data_augmentation_methods))

    def __call__(self, input_dict):
        # randint generate int x in range: a <= x <= b
        augment_method_idx = random.randint(0, len(self.data_augmentation_methods) - 1)
        augment_method = self.data_augmentation_methods[augment_method_idx]
        # print(augment_method.__class__.__name__) # debug usage
        return augment_method(input_dict)


class Mask(object):
    """
    随机掩盖其中的某些特征
    """

    def __init__(self, gamma=0.1, features_name=None):
        if features_name is None:
            features_name = []
        self.gamma = gamma
        self.features_name = features_name

    def __call__(self, input_dict):
        # make a deep copy to avoid original sequence be modified
        # copied_input_dict = copy.deepcopy(input_dict)
        # mask_nums = int(self.gamma * len(self.features_name))
        #
        # # 随机选择需要掩盖的特征
        # features_to_mask = random.sample(self.features_name, mask_nums)
        #
        # for feature in features_to_mask:
        #     # 获取特征值
        #     feature_value = copied_input_dict[feature]
        #     # 判断特征值的类型，然后进行相应的掩盖操作
        #     if feature_value.dtype == tf.string:
        #         masked_value = tf.constant(['<nan>'], dtype=tf.string)
        #     else:
        #         masked_value = tf.zeros_like(feature_value)
        #     copied_input_dict[feature] = masked_value
        # return copied_input_dict

        # 假设所有特征的batch size相同
        # batch_size = tf.shape(input_dict[self.features_name[0]])[0]
        # features_num = len(self.features_name)
        #
        # # 计算每个样本需要掩盖的特征数量
        # mask_nums_per_sample = int(self.gamma * features_num)
        #
        # # 生成随机掩码矩阵
        # mask_matrix = np.zeros((batch_size, features_num), dtype=np.float32)
        # for i in range(batch_size):
        #     mask_indices = np.random.choice(features_num, mask_nums_per_sample, replace=False)
        #     mask_matrix[i, mask_indices] = 1.0
        #
        # # 转换为Tensor
        # mask_matrix = tf.convert_to_tensor(mask_matrix, dtype=tf.float32)
        #
        # # 应用掩码
        # for i, feature in enumerate(self.features_name):
        #     feature_value = input_dict[feature]
        #     if feature_value.dtype != tf.string:
        #         # 将掩码应用于特征
        #         masked_feature_value = feature_value * (1 - mask_matrix[:, i])
        #
        #         # 将掩码后的特征值放回字典
        #         input_dict[feature] = masked_feature_value
        #
        # return input_dict

        # 假设所有特征的batch size相同
        batch_size = tf.shape(input_dict[self.features_name[0]])[0]
        features_num = len(self.features_name)

        # 计算每个样本需要掩盖的特征数量
        mask_nums_per_sample = 1
        # 随机某一位为0
        random_indices = tf.random.uniform(shape=[batch_size], minval=0, maxval=features_num, dtype=tf.int32)

        # 创建掩码矩阵
        mask_matrix = tf.one_hot(random_indices, depth=features_num, dtype=tf.int64)

        # 应用掩码
        for i, feature in enumerate(self.features_name):
            feature_value = input_dict[feature]
            if feature_value.dtype != tf.string:
                mask_vector = tf.expand_dims(1 - mask_matrix[:, i], -1)  # 将形状从[1024]变为[1024, 1]
                masked_feature_value = feature_value * mask_vector
                # 将掩码后的特征值放回字典
                input_dict[feature] = masked_feature_value

        return input_dict

from utils.tfrecord_util import read_tfrecord
import tensorflow as tf
import os
import json
import data_process.dataset_process as dp


dataset, test_dataset = dp.train_test_dataset(1024)
TYPE_DICT = {'string': tf.string, 'int64': tf.int64, 'float32': tf.float32}

# feature 文件
current_dir = os.path.dirname(__file__)
json_file = os.path.join(current_dir, 'tfrecord_feature.json')
all_features = json.load(open(json_file))

# 构建  feature_name 列表
feature_names = []
for key, value in all_features.items():
    feature_names.append(key)
feature_names.remove('label')
feature_names.remove('cvr_label')

# 将数据写入CSV
# dataset_to_csv('/Users/nowcoder/feature_data.csv', dataset, feature_names)

# 对数据做一些统计
train_count = 0
valid_count = 0
user_set = set()
item_set = set()
positive_cnt = 0
negative_cnt = 0
ios_cnt = 0
android_cnt = 0
web_cnt = 0

# for features, (label, cvr) in dataset:
#     train_count += 1024
#     # 将TensorFlow的特征转换为标准Python类型, tensorflow张量可以调用numpy()
#     features_numpy = {key: value.numpy() for key, value in features.items()}
#     label_numpy = label.numpy()
#     cvr_numpy = cvr.numpy()
#     # iter遍历dict只是返回key的迭代器，然后next取的是每一个key
#     # shape是numpy数组的方法，可以查一下，这个shape[0]获取行的个数
#     batch_size = features_numpy[next(iter(features_numpy))].shape[0]
#
#     for i in range(batch_size):
#         sample_feature = {key: features_numpy[key][i] for key in feature_names}
#         user_set.add(sample_feature["uid"][0])
#         user_set.add(sample_feature["author_uid"][0])
#         item_set.add(sample_feature["entity_id"][0])
#         item_set.update(sample_feature["hist_entity_id"])
#         item_set.update(sample_feature["expo_not_click_entity_id"])
#         if label_numpy[i][0] == 1 or cvr_numpy[i][0] == 1:
#             positive_cnt = positive_cnt + 1
#         else:
#             negative_cnt = negative_cnt + 1
#         if sample_feature['platform'][0] == b'iOS':
#             ios_cnt = ios_cnt + 1
#         elif sample_feature['platform'][0] == b'Android':
#             android_cnt = android_cnt + 1
#         elif sample_feature['platform'][0] == b'web':
#             web_cnt = web_cnt + 1
#
# print(f"用户数量:{len(user_set)}")
# print(f"物品数量:{len(item_set)}")
# print(f"正样本:{positive_cnt}")
# print(f"负样本:{negative_cnt}")
# print(f"ios:{ios_cnt}")
# print(f"android:{android_cnt}")
# print(f"web:{web_cnt}")

valid_user_set = set()
valid_item_set = set()
valid_positive_cnt = 0
valid_negative_cnt = 0
valid_ios_cnt = 0
valid_android_cnt = 0
valid_web_cnt = 0

for features, (label, cvr) in test_dataset:
    valid_count += 1024
    # 将TensorFlow的特征转换为标准Python类型, tensorflow张量可以调用numpy()
    features_numpy = {key: value.numpy() for key, value in features.items()}
    label_numpy = label.numpy()
    cvr_numpy = cvr.numpy()
    # iter遍历dict只是返回key的迭代器，然后next取的是每一个key
    # shape是numpy数组的方法，可以查一下，这个shape[0]获取行的个数
    batch_size = features_numpy[next(iter(features_numpy))].shape[0]

    for i in range(batch_size):
        sample_feature = {key: features_numpy[key][i] for key in feature_names}
        valid_user_set.add(sample_feature["uid"][0])
        valid_user_set.add(sample_feature["author_uid"][0])
        valid_item_set.add(sample_feature["entity_id"][0])
        valid_item_set.update(sample_feature["hist_entity_id"])
        valid_item_set.update(sample_feature["expo_not_click_entity_id"])
        if label_numpy[i][0] == 1 or cvr_numpy[i][0] == 1:
            valid_positive_cnt = valid_positive_cnt + 1
        else:
            valid_negative_cnt = valid_negative_cnt + 1
        if sample_feature['platform'][0] == b'iOS':
            valid_ios_cnt = valid_ios_cnt + 1
        elif sample_feature['platform'][0] == b'Android':
            valid_android_cnt = valid_android_cnt + 1
        elif sample_feature['platform'][0] == b'web':
            valid_web_cnt = valid_web_cnt + 1
print("")










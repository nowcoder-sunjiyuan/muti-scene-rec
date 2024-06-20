import data_parse
from utils.tfrecord_util import read_tfrecord
from utils.base_tool import dataset_to_csv
import tensorflow as tf
import os
import json



train_file, test_file = data_parse.mac_tain_test_file()
TYPE_DICT = {'string': tf.string, 'int64': tf.int64, 'float32': tf.float32}

# feature 文件
current_dir = os.path.dirname(__file__)
json_file = os.path.join(current_dir, 'feature.json')
all_features = json.load(open(json_file))

# 构建 feature_description, dataset
feature_description = {}
feature_names = []
for key, value in all_features.items():
    feature_description[key] = tf.io.FixedLenFeature(shape=(value[1],), dtype=TYPE_DICT[value[0]])
    feature_names.append(key)
dataset = read_tfrecord(train_file, feature_description, 1024)

# 将数据写入CSV
# dataset_to_csv('/Users/nowcoder/output.csv', dataset, feature_names)

# 对数据做一些统计
user_set = set()
item_set = set()
positive_cnt = 0
negative_cnt = 0
ios_cnt = 0
android_cnt = 0
web_cnt = 0

for features in dataset:
    # 将TensorFlow的特征转换为标准Python类型, tensorflow张量可以调用numpy()
    features_numpy = {key: value.numpy() for key, value in features.items()}
    # iter遍历dict只是返回key的迭代器，然后next取的是每一个key
    # shape是numpy数组的方法，可以查一下，这个shape[0]获取行的个数
    batch_size = features_numpy[next(iter(features_numpy))].shape[0]

    for i in range(batch_size):
        sample_feature = {key: features_numpy[key][i] for key in feature_names}
        user_set.add(sample_feature["uid"][0])
        user_set.add(sample_feature["author_uid"][0])
        item_set.add(sample_feature["entity_id"][0])
        item_set.update(sample_feature["hist_entity_id"])
        item_set.update(sample_feature["expo_not_click_entity_id"])
        if sample_feature['label'][0] == 1 or sample_feature['cvr_label'][0] == 1:
            positive_cnt = positive_cnt + 1
        else:
            negative_cnt = negative_cnt + 1
        if sample_feature['platform'][0] == b'iOS':
            ios_cnt = ios_cnt + 1
        elif sample_feature['platform'][0] == b'Android':
            android_cnt = android_cnt + 1
        elif sample_feature['platform'][0] == b'web':
            web_cnt = web_cnt + 1

print(f"用户数量:{len(user_set)}")
print(f"物品数量:{len(item_set)}")
print(f"正样本:{positive_cnt}")
print(f"负样本:{negative_cnt}")
print(f"ios:{ios_cnt}")
print(f"android:{android_cnt}")
print(f"web:{web_cnt}")











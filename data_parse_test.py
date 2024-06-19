import data_parse
from utils.tfrecord_util import read_tfrecord
from utils.base_tool import dataset_to_csv
import tensorflow as tf
import os
import json



train_file, test_file = data_parse.win_train_test_file()
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
# dataset_to_csv('', dataset, feature_names)




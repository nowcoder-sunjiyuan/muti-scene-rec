import datetime
import os
import sys
import tensorflow as tf
import json
from utils.tfrecord_util import read_tfrecord
import utils.base_tool as base_tool
import pandas as pd
import keras

TYPE_DICT = {'string': tf.string, 'int64': tf.int64, 'float32': tf.float32}


def win_train_test_file():
    start_time = datetime.datetime(2024, 5, 11, 0, 0, 0)
    train_days_num = 10
    test_days_num = 1
    data_path = '/home/web/sunjiyuan/data/essm/v1'

    train_files, test_files = [], []
    # 加载训练文件
    for i in range(train_days_num):
        cur = (start_time + datetime.timedelta(days=i)).strftime('%Y%m%d')
        if not os.path.exists(os.path.join(data_path, cur)):
            print("不存在文件夹")
            continue
        file_list = os.listdir(os.path.join(data_path, cur))
        train_files += [os.path.join(data_path, cur, fn) for fn in file_list]
    train_files.sort()

    # 加载测试文件
    test_start_time = start_time + datetime.timedelta(days=train_days_num)
    for i in range(test_days_num):
        cur = (test_start_time + datetime.timedelta(days=i)).strftime('%Y%m%d')
        file_list = os.listdir(os.path.join(data_path, cur))
        test_files += [os.path.join(data_path, cur, fn) for fn in file_list]
    test_files.sort()

    print(f"添加的文件数量：训练集 {len(train_files)} 测试集 {len(test_files)}")
    return train_files, test_files


def linux_train_test_file():
    start_time = datetime.datetime(2024, 5, 11, 0, 0, 0)
    train_days_num = 10
    test_days_num = 1
    data_path = '/opt/data/'

    train_files, test_files = [], []
    # 加载训练文件
    for i in range(train_days_num):
        cur = (start_time + datetime.timedelta(days=i)).strftime('%Y%m%d')
        if not os.path.exists(os.path.join(data_path, cur)):
            print("不存在文件夹")
            continue
        file_list = os.listdir(os.path.join(data_path, cur))
        train_files += [os.path.join(data_path, cur, fn) for fn in file_list]
    train_files.sort()

    # 加载测试文件
    test_start_time = start_time + datetime.timedelta(days=train_days_num)
    for i in range(test_days_num):
        cur = (test_start_time + datetime.timedelta(days=i)).strftime('%Y%m%d')
        file_list = os.listdir(os.path.join(data_path, cur))
        test_files += [os.path.join(data_path, cur, fn) for fn in file_list]
    test_files.sort()

    print(f"添加的文件数量：训练集 {len(train_files)} 测试集 {len(test_files)}")
    return train_files, test_files



def mac_train_test_file():
    train_files, test_files = ['/Users/nowcoder/data/2024061900.tfrecords',
                               '/Users/nowcoder/data/2024061901.tfrecords',
                               '/Users/nowcoder/data/2024061902.tfrecords',
                               '/Users/nowcoder/data/2024061903.tfrecords',
                               ], []

    print(f"添加的文件数量：训练集 {len(train_files)} 测试集 {len(test_files)}")
    return train_files, test_files


def train_test_dataset(batch_size: int):
    train_file, test_file = [], []
    if sys.platform.startswith("win"):
        train_file, test_file = win_train_test_file()
    elif sys.platform.startswith("linux"):
        train_file, test_file = linux_train_test_file()
    elif sys.platform.startswith("darwin"):
        train_file, test_file = mac_train_test_file()
    else:
        raise

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
    dataset = read_tfrecord(train_file, feature_description, batch_size, labels=["label"])
    test_dataset = read_tfrecord(test_file, feature_description, batch_size, labels=["label"])
    return dataset, test_dataset


# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('label')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds


def get_dataset_dataframe(batch_size):
    dataframe = pd.read_csv('../data/feature_data.csv')
    train_ds = df_to_dataset(dataframe, batch_size=batch_size)
    return train_ds


def build_input_tensor():
    current_dir = os.path.dirname(__file__)
    json_file = os.path.join(current_dir, 'feature.json')
    all_features = json.load(open(json_file))

    inputs = base_tool.MultiIODict({})
    for key, value in all_features.items():
        _type, _shape = TYPE_DICT[all_features[key][0]], all_features[key][1]
        inputs[key] = keras.Input(shape=(_shape,), name=key, dtype=_type)
    return inputs

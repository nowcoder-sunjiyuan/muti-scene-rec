import datetime
import os
import sys
import time

import tensorflow as tf
import json
from utils.tfrecord_util import read_tfrecord
import utils.base_tool as base_tool
import pandas as pd
import keras
import data_process.data_augmentation as da
import copy
import data_process.feature_process as fp
import random
from models.FeatureEmbModel import FeatureEmbModel

TYPE_DICT = {'string': tf.string, 'int64': tf.int64, 'float32': tf.float32}


def win_train_test_file():
    start_time = datetime.datetime(2024, 7, 1, 0, 0, 0)
    train_days_num = 14
    test_days_num = 2
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
    start_time = datetime.datetime(2024, 7, 1, 0, 0, 0)
    train_days_num = 14
    test_days_num = 2
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
    json_file = os.path.join(current_dir, 'tfrecord_feature.json')
    all_features = json.load(open(json_file))

    # 构建 feature_description, dataset
    feature_description = {}
    feature_names = []
    for key, value in all_features.items():
        feature_description[key] = tf.io.FixedLenFeature(shape=(value[1],), dtype=TYPE_DICT[value[0]])
        feature_names.append(key)
    dataset = read_tfrecord(train_file, feature_description, batch_size, labels=["label", "cvr_label"])
    test_dataset = read_tfrecord(test_file, feature_description, batch_size, labels=["label", "cvr_label"])
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
    json_file = os.path.join(current_dir, 'input_feature.json')
    all_features = json.load(open(json_file))

    inputs = base_tool.MultiIODict({})
    for key, value in all_features.items():
        _type, _shape = TYPE_DICT[all_features[key][0]], all_features[key][1]
        inputs[key] = keras.Input(shape=(_shape,), name=key, dtype=_type)
    return inputs


class DatasetProcess:

    def __init__(self):
        # 加载特征dict
        current_dir = os.path.dirname(__file__)
        json_file = os.path.join(current_dir, 'input_feature_cf.json')
        self.cf_features = json.load(open(json_file))

        # 特征表示
        self.feature_description = {}
        self.features_name = []
        for key, value in self.cf_features.items():
            self.feature_description[key] = tf.io.FixedLenFeature(shape=(value[1],), dtype=TYPE_DICT[value[0]])
            self.features_name.append(key)
        self.label_description = {"label": tf.io.FixedLenFeature(shape=(1,), dtype="int64")}
        self.feature_parse_model = FeatureEmbModel()

        # 数据增强方案
        self.augmentations = {
            "mask": da.Mask(features_name=self.features_name),
            "random": da.Random(features_name=self.features_name),
        }
        self.base_transform = self.augmentations["random"]

    def example_generator_cl(self, tfrecord_filenames, batch_size):
        dataset = tf.data.TFRecordDataset(tfrecord_filenames)
        dataset = dataset.batch(batch_size)

        def filter_batch_size(raw_record):
            # 检查当前批次的大小是否等于batch_size
            return tf.equal(tf.shape(raw_record)[0], batch_size)

        # 只保留大小等于batch_size的批次，这个是为了舍弃最后那几个batch_size不符合要求的部分
        dataset = dataset.filter(filter_batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        for raw_record in dataset:
            input_dict = tf.io.parse_example(raw_record, self.feature_description)  # dict

            # 对里面每个特征都进行一个预处理
            target, parse_range_dict = self.feature_parse_model(input_dict)

            target_pos = self.base_transform(target)
            target_neg = self._generate_neg(target, parse_range_dict, batch_size)
            label_dict = tf.io.parse_example(raw_record, self.label_description)
            yield target, target_pos, target_neg, label_dict

    def create_dataset_cl(self, tfrecord_filenames, batch_size):

        # dataset = tf.data.TFRecordDataset(tfrecord_filenames)
        # dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        # for raw_record in dataset:
        #     input_dict = tf.io.parse_example(raw_record, self.feature_description)
        #     # input_dict, target_pos, target_neg = self.process_single_example(example_proto)
        #     # 对里面每个特征都进行一个预处理
        #     parse_input_dict, parse_range_dict = self.feature_parse_model(input_dict)
        #
        #     time1 = time.time()
        #     target_pos = self.base_transform(parse_input_dict)
        #     time2 = time.time()
        #     target_neg = self._generate_neg(parse_input_dict, parse_range_dict, batch_size)
        #     time3 = time.time()
        #     label_dict = tf.io.parse_example(raw_record, self.label_description)
        #     print(f"pos：{time2 - time1:.2f}s，neg: {time3 - time2:.2f}")

        # # 添加其他所需操作，例如batching和prefetching

        # self.example_generator(tfrecord_filenames, batch_size)

        output_signature = (
            {k: tf.TensorSpec(shape=(batch_size, 1), dtype="int64") for k, v in self.cf_features.items()},
            {k: tf.TensorSpec(shape=(batch_size, 1), dtype="int64") for k, v in self.cf_features.items()},
            {k: tf.TensorSpec(shape=(batch_size, 1), dtype="int64") for k, v in self.cf_features.items()},
            {"label": tf.TensorSpec(shape=(batch_size, 1), dtype="int64")}
        )
        dataset = tf.data.Dataset.from_generator(
            lambda: self.example_generator_cl(tfrecord_filenames, batch_size),
            output_signature=output_signature
        )
        return dataset

    def example_generator_normal(self, tfrecord_filenames, batch_size):
        dataset = tf.data.TFRecordDataset(tfrecord_filenames)
        dataset = dataset.batch(batch_size)

        def filter_batch_size(raw_record):
            # 检查当前批次的大小是否等于batch_size
            return tf.equal(tf.shape(raw_record)[0], batch_size)

        # 只保留大小等于batch_size的批次，这个是为了舍弃最后那几个batch_size不符合要求的部分
        dataset = dataset.filter(filter_batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        for raw_record in dataset:
            input_dict = tf.io.parse_example(raw_record, self.feature_description)

            # 对里面每个特征都进行一个预处理
            target, parse_range_dict = self.feature_parse_model(input_dict)
            label_dict = tf.io.parse_example(raw_record, self.label_description)
            yield target, label_dict

    def create_dataset_normal(self, tfrecord_filenames, batch_size):
        """
        创建数据集，普通方式
        :param tfrecord_filenames:
        :param batch_size:
        :return:
        """

        dataset = tf.data.TFRecordDataset(tfrecord_filenames)
        dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        # for raw_record in dataset:
        #     input_dict = tf.io.parse_example(raw_record, self.feature_description)
        #     # input_dict, target_pos, target_neg = self.process_single_example(example_proto)
        #     # 对里面每个特征都进行一个预处理
        #     parse_input_dict, parse_range_dict = self.feature_parse_model(input_dict)
        #     label_dict = tf.io.parse_example(raw_record, self.label_description)

        # # 添加其他所需操作，例如batching和prefetching

        # self.example_generator(tfrecord_filenames, batch_size)

        output_signature = (
            {k: tf.TensorSpec(shape=(batch_size, 1), dtype="int64") for k, v in self.cf_features.items()},
            {"label": tf.TensorSpec(shape=(batch_size, 1), dtype="int64")}
        )
        dataset = tf.data.Dataset.from_generator(
            lambda: self.example_generator_normal(tfrecord_filenames, batch_size),
            output_signature=output_signature
        )
        return dataset

    def _generate_neg(self, input_dict, input_range_dict, batch_size):
        """
        产生负样本, 我们将每个特征都随机选择一个值
        """
        copied_input_dict = {}
        for key in input_dict:
            # 对于input_dict中的每个键，生成一个范围在0到input_range_dict[key]之间的随机整数张量
            copied_input_dict[key] = fp.generate_random_int_tensor_tf(batch_size, 0, input_range_dict[key])
        return copied_input_dict

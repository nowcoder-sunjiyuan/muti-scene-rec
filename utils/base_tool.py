import numpy as np
import csv


def decode_if_bytes(value):
    # 如果值是字节串，解码为字符串
    if isinstance(value, bytes):
        return value.decode('utf-8')
    # 如果值是列表，对列表中的每个元素进行检查
    elif isinstance(value, np.ndarray) or isinstance(value, list):
        return [item.decode('utf-8') if isinstance(item, bytes) else item for item in value]
    # 如果不是上述两种情况，直接返回原值
    else:
        return value


def dataset_to_csv(csv_url, dataset, feature_names):
    with open(csv_url, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=feature_names)
        writer.writeheader()
        for features in dataset:
            # 将TensorFlow的特征转换为标准Python类型, tensorflow张量可以调用numpy()
            features_numpy = {key: value.numpy() for key, value in features.items()}
            # iter遍历dict只是返回key的迭代器，然后next取的是每一个key
            # shape是numpy数组的方法，可以查一下，这个shape[0]获取行的个数
            batch_size = features_numpy[next(iter(features_numpy))].shape[0]

            print("----------还在遍历中---------")
            for i in range(batch_size):
                sample_feature = {key: decode_if_bytes(features_numpy[key][i]) for key in feature_names}
                writer.writerow(sample_feature)


class MultiIODict(dict):
    """
    因为比较懒，不想写很多行，所以写了这个类，支持单行多输入输出
    可以在[]中传入多个 key，返回一个 list
    在为字典设置值时也可以传入多个 key 和 value，要求 key 和 value 的数量一致
    Example:
        a = MultiIODict()
        a["a", "b"] = 1, 2
        print(a["a", "b"])
    """

    def __getitem__(self, item):
        if not isinstance(item, tuple):
            return super().__getitem__(item)
        res = []
        for key in item:
            res.append(super().__getitem__(key))
        return res

    def __setitem__(self, key, value):
        if not isinstance(key, tuple):
            super().__setitem__(key, value)
            return
        assert len(key) == len(value), "key 和 value 数量不一致"
        for each in zip(key, value):
            super().__setitem__(*each)


def read_vocabulary(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

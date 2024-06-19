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
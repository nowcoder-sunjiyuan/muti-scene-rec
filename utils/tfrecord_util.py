import tensorflow as tf


def read_tfrecord(
        filenames,
        feature_description,
        batch_size,
        num_parallel_calls=8,
        prefetch_num=1,
        labels=tuple(),
        use_weight=False,
        weight_name=None
):
    def _parse_example(example_proto):
        features = tf.io.parse_single_example(example_proto, feature_description)
        _labels = []
        for label in labels:
            _labels.append(features.pop(label))
        if use_weight:
            if not weight_name:
                raise AttributeError('权重列的名称不能为空！')
            weight = features.pop(weight_name)
            return features, tuple(_labels), weight
        return features, tuple(_labels)

    def _input():
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(_parse_example, num_parallel_calls=num_parallel_calls)
        dataset = dataset.batch(batch_size)
        if prefetch_num > 0:
            dataset = dataset.prefetch(buffer_size=batch_size * prefetch_num)
        return dataset
    return _input()

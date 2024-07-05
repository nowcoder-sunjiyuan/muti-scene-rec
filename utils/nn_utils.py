from data_process import data_process
import numpy as np
import tensorflow as tf
import keras
from keras import activations, initializers, regularizers, constraints

from keras import layers, Layer

l2_reg = 0.0000025
L2REG = keras.regularizers.L2(l2_reg)


# 将输入经过 string_lookup再经过 embedding 转换
def string_lookup_embedding(inputs, voc_list, embedding_dimension=16, embedding_regularizer=L2REG,
                            embedding_initializer='glorot_normal', name=''):
    string_lookup_layer = layers.StringLookup(vocabulary=voc_list,
                                              name=f"{name}_lookup_layer")
    embedding_layer = layers.Embedding(input_dim=len(string_lookup_layer.get_vocabulary()) + 1,
                                       output_dim=embedding_dimension,
                                       embeddings_regularizer=embedding_regularizer,
                                       embeddings_initializer=embedding_initializer,
                                       name=f"{name}_embedding_layer")

    shared = isinstance(inputs, (tuple, list))
    if shared:
        res = []
        for each in inputs:
            embedding_output = _embedding_res_check_reshape(embedding_layer(string_lookup_layer(each)))
            res.append(embedding_output)
        return res
    else:
        embedding_output = _embedding_res_check_reshape(embedding_layer(string_lookup_layer(inputs)))
        return embedding_output


# 将输入经过 hash再经过 embedding
def hash_lookup_embedding(inputs, num_bins, embedding_dimension=16, embedding_regularizer=L2REG,
                          embedding_initializer='glorot_normal', name=''):
    hash_layer = layers.Hashing(num_bins=num_bins, name=f"{name}_hash_layer")
    embedding_layer = layers.Embedding(input_dim=num_bins,
                                       output_dim=embedding_dimension,
                                       embeddings_initializer=embedding_initializer,
                                       embeddings_regularizer=embedding_regularizer)

    shared = isinstance(inputs, (tuple, list))
    if shared:
        res = []
        for each in inputs:
            embedding_output = _embedding_res_check_reshape(embedding_layer(hash_layer(each)))
            res.append(embedding_output)
        return res
    else:
        embedding_output = _embedding_res_check_reshape(embedding_layer(hash_layer(inputs)))
        return embedding_output


def integer_lookup_embedding(inputs, voc_list, embedding_dimension=16, embedding_regularizer=L2REG,
                             embedding_initializer='glorot_normal', name=''):
    integer_lookup_layer = layers.IntegerLookup(vocabulary=voc_list,
                                                name=f"{name}_lookup_layer")
    embedding_layer = layers.Embedding(input_dim=len(integer_lookup_layer.get_vocabulary()) + 1,
                                       output_dim=embedding_dimension,
                                       embeddings_regularizer=embedding_regularizer,
                                       embeddings_initializer=embedding_initializer,
                                       name=f"{name}_embedding_layer")
    shared = isinstance(inputs, (tuple, list))
    if shared:
        res = []
        for each in inputs:
            embedding_output = _embedding_res_check_reshape(embedding_layer(integer_lookup_layer(each)))
            res.append(embedding_output)
        return res
    else:
        embedding_output = _embedding_res_check_reshape(embedding_layer(integer_lookup_layer(inputs)))
        return embedding_output


# 对embedding结果进行检测和reshape
def _embedding_res_check_reshape(embedding_res):
    if embedding_res.shape[1] == 1:
        # dataset batch 后经过 lookup 和 embedding  (none, 1, 16)
        reshape_layer = layers.Reshape((embedding_res.shape[-1],))
        embedding_res = reshape_layer(embedding_res)
        return embedding_res
        # embedding_output = tf.reshape(embedding_output, [-1, embedding_output.shape[-1]])
    else:
        return embedding_res


class ExpandDimsLayer(Layer):
    def __init__(self, axis, **kwargs):
        super(ExpandDimsLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)

    def compute_output_shape(self, input_shape):
        if self.axis < 0:
            self.axis = len(input_shape) + self.axis + 1
        return input_shape[:self.axis] + (1,) + input_shape[self.axis:]


class ReduceSumLayer(Layer):
    def __init__(self, axis, **kwargs):
        super(ReduceSumLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=self.axis)

    def compute_output_shape(self, input_shape):
        return input_shape[:self.axis] + input_shape[self.axis + 1:]


class ReduceMeanWithMask(Layer):
    """
    对输入的inputs序列数据，根据真实长度，求平均
    """

    def __init__(self, max_len, elem_type=tf.float32, **kwargs):
        super(ReduceMeanWithMask, self).__init__(**kwargs)
        self.max_len = max_len
        self.elem_type = elem_type

    def call(self, inputs, valid_length):
        # 给每个序列生成掩码
        mask = tf.sequence_mask(valid_length, maxlen=self.max_len, dtype=self.elem_type)
        mask = tf.reshape(mask, shape=(-1, self.max_len, 1))
        masked_embeddings = inputs * mask
        summed = tf.reduce_sum(masked_embeddings, axis=1)
        return summed / tf.cast(valid_length, dtype=tf.float32)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]


class FullyConnectedTower(keras.layers.Layer):
    """
    若干个全连接层组成的塔
    """

    def __init__(self,
                 tower_units: list,
                 tower_name: str,
                 hidden_activation,
                 output_activation=None,
                 regularizer=keras.regularizers.L2(0.00001),
                 use_bn=True,
                 dropout=0.0,
                 seed=2023,
                 **kwargs):
        self.tower_units = tower_units
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.regularizer = regularizer
        self.use_bn = use_bn
        self.tower_name = tower_name
        self.seed = seed
        self.dropout = dropout

        self.kernels = None
        self.activations = None
        if self.use_bn:
            self.batch_normalizations = None
        self.dropout_layers = None

        super(FullyConnectedTower, self).__init__(**kwargs)

    def build(self, input_shape):

        self.activations = [activations.get(self.hidden_activation) for _ in range(len(self.tower_units) - 1)] + \
                           [activations.get(self.output_activation)]

        self.kernels = [layers.Dense(self.tower_units[i],
                                     kernel_initializer=keras.initializers.glorot_normal(seed=self.seed),
                                     kernel_regularizer=self.regularizer) for i in range(len(self.tower_units))]

        if self.use_bn:
            self.batch_normalizations = [keras.layers.BatchNormalization() for i in range(len(self.tower_units))]

        if self.dropout > 0.0:
            self.dropout_layers = [keras.layers.Dropout(self.dropout) for i in range(len(self.tower_units))]

        super(FullyConnectedTower, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        tower_input = inputs
        for i in range(len(self.tower_units)):
            cur = self.kernels[i](tower_input)
            if self.use_bn:
                cur = self.batch_normalizations[i](cur, training=training)
            if self.activations[i] is not None:
                cur = self.activations[i](cur)
            if self.dropout > 0.0:
                cur = self.dropout_layers[i](cur)
            tower_input = cur

        return tower_input

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        final_unit = self.tower_units[-1]
        output_shape[-1] = final_unit
        output_shape = tuple(output_shape)
        return output_shape

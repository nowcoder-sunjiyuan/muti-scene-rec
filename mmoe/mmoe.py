import keras.src.layers

from data_process import data_process
import numpy as np
import tensorflow as tf

from keras import layers
"""
dataset就像，一个巨大的 dict
key 是特征名称
value 是个列表，每个
"""
dataset = data_process.train_test_dataset(1024)

l2_reg = 0.0000025
L2REG = keras.regularizers.L2(l2_reg)

# 字符串类特征
gender_lookup_layer = layers.StringLookup(
    vocabulary=["<nan>", "其他", "男", "女"],
    name='gender_look_up'
)
gender_embedding_layer = layers.Embedding(
    input_dim=len(gender_lookup_layer.get_vocabulary()) + 1,
    output_dim=16,
    embeddings_regularizer=L2REG,
    embeddings_initializer='glorot_normal',
    name="gender_embedding_layer"
)
# dataset.map(lambda x, y: )
lookup_output = gender_lookup_layer()
embedding_output = gender_embedding_layer(lookup_output)


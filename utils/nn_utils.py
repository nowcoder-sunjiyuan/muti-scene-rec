from data_process import data_process
import numpy as np
import tensorflow as tf

from keras import layers


# 将输入经过 string_lookup再经过 embedding 转换
def string_lookup_embedding(inputs, voc_list, embedding_dimension, embedding_regularizer, embedding_initializer,
                            name):
    string_lookup_layer = layers.StringLookup(vocabulary=voc_list,
                                              name=f"{name}_lookup_layer")
    embedding_layer = layers.Embedding(input_dim=len(string_lookup_layer.get_vocabulary()) + 1,
                                       output_dim=embedding_dimension,
                                       embeddings_regularizer=embedding_regularizer,
                                       embeddings_initializer=embedding_initializer,
                                       name=f"{name}_embedding_layer")
    embedding_output = embedding_layer(string_lookup_layer(inputs))

    if embedding_output.shape[1] == 1:
        # dataset batch 后经过 lookup 和 embedding  (none, 1, 16)
        embedding_output = tf.reshape(embedding_output, [-1, embedding_output.shape[-1]])
    return embedding_output


# 将输入经过 hash再经过 embedding
def hash_lookup_embedding(inputs, num_bins, embedding_dimension, embedding_regularizer, embedding_initializer, name):
    hash_layer = layers.Hashing(num_bins=num_bins, name=f"{name}_hash_layer")
    embedding_layer = layers.Embedding(input_dim=num_bins,
                                       output_dim=embedding_dimension,
                                       embeddings_initializer=embedding_initializer,
                                       embeddings_regularizer=embedding_regularizer)
    embedding_output = embedding_layer(hash_layer(inputs))
    if embedding_output.shape[1] == 1:
        embedding_output = tf.reshape(embedding_output, [-1, embedding_output.shape[-1]])
    return embedding_output


def reduce_mean_with_mask(inputs, valid_length, max_len, elem_type=tf.float32):
    # mask 是由 length 和 max_len 通过函数得到的
    embeddings = inputs
    mask = get_mask(valid_length, max_len, elem_type)
    masked_embeddings = embeddings * mask
    summed = tf.reduce_sum(masked_embeddings, axis=1)
    return summed / tf.cast(valid_length, dtype=tf.float32)


def get_mask(valid_length, max_len, elem_type=tf.float32):
    return tf.reshape(tf.sequence_mask(valid_length, maxlen=max_len, dtype=elem_type), shape=(-1, max_len, 1))

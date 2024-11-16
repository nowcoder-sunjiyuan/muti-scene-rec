import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer, Embedding


class SelfAttention(Layer):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

    def call(self, inputs):
        # inputs 的形状: (batch_size, seq_length, embed_dim)
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        # 计算缩放点积注意力
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)

        # 计算注意力输出
        attention = tf.matmul(weights, value)
        output = self.combine_heads(attention)
        return output

# # 示例使用
# batch_size = 256
# seq_length = 50
# embed_dim = 32
#
# # 假设输入数据是整数序列
# inputs = tf.random.uniform((batch_size, seq_length), maxval=100, dtype=tf.int32)
#
# # 嵌入层将输入转换为嵌入向量
# embedding_layer = Embedding(input_dim=100, output_dim=embed_dim)
# embedded_inputs = embedding_layer(inputs)  # (batch_size, seq_length, embed_dim)
#
# # 自注意力层
# self_attention_layer = SelfAttention(embed_dim)
# outputs = self_attention_layer(embedded_inputs)
#
# print(outputs.shape)  # 期望输出形状: (256, 50, 32)

import keras
import tensorflow as tf
from keras import layers, Layer
import numpy as np
from models.DCN import DynamicConv1D


class SelfAttention(Layer):
    """
    输入是 (none, seq_length, embed_dim)
    """

    def __init__(self, emb_dim):
        """
        :param emb_dim: 这个是输入的维度，也是输出的维度，最后就是为了产生一个同纬度的高级表达
        """
        super(SelfAttention, self).__init__()
        self.embed_dim = emb_dim
        self.query_dense = layers.Dense(emb_dim)  # (embed_dim * embed_dim + embed _dim)参数量的dense
        self.key_dense = layers.Dense(emb_dim)
        self.value_dense = layers.Dense(emb_dim)
        self.combine_heads = layers.Dense(emb_dim)

    def call(self, inputs):
        # inputs 的形状: (batch_size, seq_length, embed_dim)
        query = self.query_dense(inputs)  # (batch_size, seq_length, query_embed_dim)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        # 计算缩放点积注意力
        score = tf.matmul(query, key, transpose_b=True)  # (batch_size, seq_length, seq_length)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)  # (batch_size, seq_length, seq_length)

        # 计算注意力输出
        attention = tf.matmul(weights, value)  # (batch, seq_length, embed_dim)
        output = self.combine_heads(attention)  # (batch, seq_length, embed_dim)
        return output


class MultiHeadAttention(Layer):
    def __init__(self, emb_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.emb_dim = emb_dim
        assert emb_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.head_dim = emb_dim // num_heads  # 每个头的维度是原来的维度除以头的数量

        self.query_dense = layers.Dense(emb_dim)
        self.key_dense = layers.Dense(emb_dim)
        self.value_dense = layers.Dense(emb_dim)
        self.combine_heads = layers.Dense(emb_dim)
        self.attention_weights = None

    def split_heads(self, x):
        batch_size = tf.shape(x)[0]
        # 重塑成 (batch_size, seq_length, num_heads, head_dim)
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        # 转置为 (batch_size, num_heads, seq_length, head_dim)
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, mask=None):
        batch_size = tf.shape(inputs)[0]

        # 线性变换
        query = self.query_dense(inputs)  # batch_size, seq_length, embed_dim
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        # 分头
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        # 计算注意力分数
        score = tf.matmul(query, key, transpose_b=True)  # (batch_size, num_heads, seq_length, seq_length)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        if mask is not None:
            scaled_score *= mask
        weights = tf.nn.softmax(scaled_score)

        self.attention_weights = weights

        # 计算注意力的输出
        attention = tf.matmul(weights, value)  # (batch_size, num_heads, seq_length, emb_dim)

        # 合并多头的结果
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])  # (batch_size, seq_length, num_heads, head_dim)
        attention = tf.reshape(attention, (batch_size, -1, self.emb_dim))  # (batch_size, seq_length, emb_dim)

        output = self.combine_heads(attention)
        return output


class EncoderLayerSelfAtt(Layer):
    def __init__(self, emb_dim, ff_dim, dropout_rate=0.1):
        super(EncoderLayerSelfAtt, self).__init__()
        self.self_attention = SelfAttention(emb_dim)

        self.layer_norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = keras.layers.Dropout(dropout_rate)
        self.dropout2 = keras.layers.Dropout(dropout_rate)

        # 点式前馈网络，就是transformer里面那个FC
        self.ffn = keras.Sequential([
            keras.layers.Dense(ff_dim, activation='relu'),
            keras.layers.Dropout(dropout_rate)
        ])

    def call(self, inputs, training):
        attention_output = self.self_attention(inputs)
        attention_output = self.dropout1(attention_output, training=training)

        output1 = self.layer_norm1(attention_output + inputs)  # 前馈网络输入前的residual和norm，残差链接和层归一化

        ffn_output = self.ffn(output1)
        ffn_output = self.dropout2(ffn_output, training=training)
        output2 = self.layer_norm2(output1 + ffn_output)  # fc输出也有一个residual和norm
        return output2


class EncoderLayerMultiHeadAtt(Layer):
    def __init__(self, emb_dim, ff_dim, dropout_rate=0.1):
        super(EncoderLayerMultiHeadAtt, self).__init__()
        self.attention = MultiHeadAttention(emb_dim, 4)

        self.layer_norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = keras.layers.Dropout(dropout_rate)
        self.dropout2 = keras.layers.Dropout(dropout_rate)

        # 点式前馈网络，就是transformer里面那个FC
        self.ffn = keras.Sequential([
            keras.layers.Dense(ff_dim, activation='relu'),
            keras.layers.Dropout(dropout_rate)
        ])

    def attention_weights(self):
        return self.attention.attention_weights

    def call(self, inputs, training):
        attention_output = self.attention(inputs)
        attention_output = self.dropout1(attention_output, training=training)

        output1 = self.layer_norm1(attention_output + inputs)  # 前馈网络输入前的residual和norm，残差链接和层归一化

        ffn_output = self.ffn(output1)
        ffn_output = self.dropout2(ffn_output, training=training)
        output2 = self.layer_norm2(output1 + ffn_output)  # fc输出也有一个residual和norm
        return output2


def create_time_en_mask(seq_len, time_en):
    # 创建一个递增的时间增强权重
    weights = np.array([time_en ** i for i in range(seq_len)])

    # 扩展为二维矩阵，每行都是相同的权重
    mask = np.tile(weights, (seq_len, 1))

    # 转换为 TensorFlow 张量
    mask = tf.convert_to_tensor(mask, dtype=tf.float32)
    mask = tf.expand_dims(mask, 0)
    mask = tf.expand_dims(mask, 0)

    return mask


class EncoderLayerTimeEn(Layer):
    def __init__(self, emb_dim, ff_dim, seq_len, time_en=1.21, dropout_rate=0.1):
        super(EncoderLayerTimeEn, self).__init__()
        self.attention = MultiHeadAttention(emb_dim, 4)

        self.layer_norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = keras.layers.Dropout(dropout_rate)
        self.dropout2 = keras.layers.Dropout(dropout_rate)

        # 点式前馈网络，就是transformer里面那个FC
        self.ffn = keras.Sequential([
            keras.layers.Dense(ff_dim, activation='relu'),
            keras.layers.Dropout(dropout_rate)
        ])

        # 预生成掩码
        self.cached_mask = create_time_en_mask(seq_len, time_en)

    def attention_weights(self):
        return self.attention.attention_weights

    def call(self, inputs, training):
        seq_len = tf.shape(inputs)[1]
        mask = self.cached_mask[:, :, :seq_len, :seq_len]  # 根据实际序列长度切片

        attention_output = self.attention(inputs, mask)
        attention_output = self.dropout1(attention_output, training=training)

        output1 = self.layer_norm1(attention_output + inputs)  # 前馈网络输入前的residual和norm，残差链接和层归一化

        ffn_output = self.ffn(output1)
        ffn_output = self.dropout2(ffn_output, training=training)
        output2 = self.layer_norm2(output1 + ffn_output)  # fc输出也有一个residual和norm
        return output2


class EncoderLayerConvTimeEn(Layer):
    def __init__(self, emb_dim, ff_dim, seq_len, time_en=1.21, dropout_rate=0.1):
        super(EncoderLayerConvTimeEn, self).__init__()
        self.attention = MultiHeadAttention(emb_dim, 4)
        self.conv = DynamicConv1D(input_size=emb_dim, kernel_size=3, num_heads=8)

        self.layer_norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = keras.layers.Dropout(dropout_rate)
        self.dropout2 = keras.layers.Dropout(dropout_rate)

        # 点式前馈网络，就是transformer里面那个FC
        self.ffn = keras.Sequential([
            keras.layers.Dense(ff_dim, activation='relu'),
            keras.layers.Dropout(dropout_rate)
        ])

        # 预生成掩码
        self.cached_mask = create_time_en_mask(seq_len, time_en)

    def call(self, inputs, training):
        seq_len = tf.shape(inputs)[1]
        mask = self.cached_mask[:, :, :seq_len, :seq_len]  # 根据实际序列长度切片

        attention_output = self.attention(inputs, mask)
        attention_output = self.dropout1(attention_output, training=training)

        conv_output = self.conv(inputs)
        output1 = self.layer_norm1(conv_output + attention_output + inputs)  # 前馈网络输入前的residual和norm，残差链接和层归一化

        ffn_output = self.ffn(output1)
        ffn_output = self.dropout2(ffn_output, training=training)
        output2 = self.layer_norm2(output1 + ffn_output)  # fc输出也有一个residual和norm
        return output2



class TrainableTimeMask(keras.layers.Layer):
    def __init__(self, max_seq_len, num_heads, init_value=1.2, ** kwargs):
        super().__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.init_value = init_value

    def build(self, input_shape):
        # 可训练参数矩阵（时间衰减系数）
        self.alpha = self.add_weight(
            name='alpha_matrix',
            shape=(self.max_seq_len, self.max_seq_len),
            initializer=tf.keras.initializers.Constant(
                value=np.log(self.init_value) *
                      np.tri(self.max_seq_len).T[::-1]  # 下三角初始化
            ),
            trainable=True
        )

        # 动态缩放系数（参考LaViT的混合注意力机制）
        self.beta = self.add_weight(
            name='beta',
            shape=(1,),
            initializer='ones',
            trainable=True
        )

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]

        # 动态矩阵切片
        mask = tf.math.exp(self.alpha[:seq_len, :seq_len])  # 保证正数
        mask = tf.linalg.band_part(mask, -1, 0)  # 保持下三角结构

        # 多头注意力适配（参考STD-MAE的空间解耦机制）
        mask = tf.reshape(
            mask,
            [1, 1, seq_len, seq_len]
        ) * self.beta  # (1, 1, seq_len, seq_len)

        return mask

    def get_config(self):
        return {
            'max_seq_len': self.max_seq_len,
            'num_heads': self.num_heads,
            'init_value': self.init_value
        }


if __name__ == "__main__":
    # 示例使用
    batch_size = 256
    seq_length = 50
    embed_dim = 32

    # 假设输入数据是整数序列
    inputs = tf.random.uniform((batch_size, seq_length), maxval=100, dtype=tf.int32)
    # 嵌入层将输入转换为嵌入向量
    embedding_layer = layers.Embedding(input_dim=100, output_dim=embed_dim)
    embedded_inputs = embedding_layer(inputs)  # (batch_size, seq_length, embed_dim)

    encoder = EncoderLayerTimeEn(embed_dim, 32, seq_length)
    outputs = encoder(embedded_inputs, training=False)
    print(outputs.shape)  # 期望输出形状: (256, 50, 32)

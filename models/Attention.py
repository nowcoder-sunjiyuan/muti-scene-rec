import keras
import tensorflow as tf
from keras import layers, Layer


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
        self.query_dense = layers.Dense(emb_dim)  # (embed_dim * embed_dim + embed_dim)参数量的dense
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
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.head_dim = embed_dim // num_heads  # 每个头的维度是原来的维度除以头的数量

        self.query_dense = layers.Dense(emb_dim)
        self.key_dense = layers.Dense(emb_dim)
        self.value_dense = layers.Dense(emb_dim)
        self.combine_heads = layers.Dense(emb_dim)

    def split_heads(self, x):
        batch_size = tf.shape(x)[0]
        # 重塑成 (batch_size, seq_length, num_heads, head_dim)
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        # 转置为 (batch_size, num_heads, seq_length, head_dim)
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
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
        weights = tf.nn.softmax(scaled_score)

        # 计算注意力的输出
        attention = tf.matmul(weights, value)  # (batch_size, num_heads, seq_length, emb_dim)

        # 合并多头的结果
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])  # (batch_size, seq_length, num_heads, head_dim)
        attention = tf.reshape(attention, (batch_size, -1, self.emb_dim))  # (batch_size, seq_length, emb_dim)

        output = self.combine_heads(attention)
        return output


class EncoderLayer(Layer):
    def __init__(self, emb_dim, ff_dim, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
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


# 示例使用
batch_size = 256
seq_length = 50
embed_dim = 32

# 假设输入数据是整数序列
inputs = tf.random.uniform((batch_size, seq_length), maxval=100, dtype=tf.int32)

# 嵌入层将输入转换为嵌入向量
embedding_layer = layers.Embedding(input_dim=100, output_dim=embed_dim)
embedded_inputs = embedding_layer(inputs)  # (batch_size, seq_length, embed_dim)

# 自注意力层
#self_attention_layer = SelfAttention(embed_dim)
#outputs = self_attention_layer(embedded_inputs)

encoder = EncoderLayer(embed_dim, 32)
outputs = encoder(embedded_inputs, training=False)
print(outputs.shape)  # 期望输出形状: (256, 50, 32)

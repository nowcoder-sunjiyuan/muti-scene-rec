import tensorflow as tf


class DynamicConv1D(tf.keras.layers.Layer):
    def __init__(self, input_size, kernel_size=1, padding='same', num_heads=1,
                 weight_dropout=0.0, weight_softmax=False, bias=False, conv_bias=False,
                 query_size=None, in_proj=False, **kwargs):
        super(DynamicConv1D, self).__init__(**kwargs)
        self.input_size = input_size
        self.query_size = input_size if query_size is None else query_size
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.weight_dropout = weight_dropout
        self.weight_softmax = weight_softmax
        self.in_proj = in_proj
        self.padding = padding

        if in_proj:
            self.weight_linear = tf.keras.layers.Dense(self.input_size + num_heads * kernel_size)
        else:
            self.weight_linear = tf.keras.layers.Dense(num_heads * kernel_size, use_bias=bias)

        if conv_bias:
            self.conv_bias = self.add_weight(shape=(input_size,), initializer='zeros', trainable=True)
        else:
            self.conv_bias = None

    def call(self, x, query=None):
        if query is None:
            query = x

        T, B, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        H = self.num_heads
        R = C // H

        if self.in_proj:
            proj = self.weight_linear(x)
            x = proj[:, :, :self.input_size]
            weight = proj[:, :, self.input_size:].reshape((T * B * H, -1))
        else:
            weight = self.weight_linear(query)
            weight = tf.reshape(weight, (T * B * H, -1))

        if self.weight_softmax:
            weight = tf.nn.softmax(weight, axis=-1)

        weight = tf.nn.dropout(weight, rate=self.weight_dropout)

        x_unfold = tf.image.extract_patches(
            images=tf.expand_dims(x, axis=0),
            sizes=[1, self.kernel_size, 1, 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding=self.padding.upper()
        )
        x_unfold = tf.reshape(x_unfold, (T * B * H, R, self.kernel_size))

        output = tf.matmul(x_unfold, tf.expand_dims(weight, -1))
        output = tf.reshape(output, (T, B, C))

        if self.conv_bias is not None:
            output += self.conv_bias

        return output


if __name__ == '__main__':
    # 示例用法
    input_tensor = tf.random.normal([50, 32, 128])  # T x B x C
    dynamic_conv = DynamicConv1D(input_size=128, kernel_size=3, num_heads=4)
    output_tensor = dynamic_conv(input_tensor)
    print(output_tensor.shape)

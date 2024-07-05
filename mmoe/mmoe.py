import keras.src.ops

import feature_representation.feature_representation as fr
from keras import layers, InputSpec
from keras import activations, initializers, regularizers, constraints, optimizers
import tensorflow as tf
import utils.nn_utils as nn
import datetime
from data_process import data_process


class MMoE(layers.Layer):
    """
    Multi-gate Mixture-of-Experts model.
    """

    def __init__(self,
                 units,
                 num_experts,
                 num_tasks,
                 use_expert_bias=True,
                 use_gate_bias=True,
                 expert_activation='relu',
                 gate_activation='softmax',
                 expert_bias_initializer='zeros',
                 gate_bias_initializer='zeros',
                 expert_bias_regularizer=None,
                 gate_bias_regularizer=None,
                 expert_bias_constraint=None,
                 gate_bias_constraint=None,
                 expert_kernel_initializer='VarianceScaling',
                 gate_kernel_initializer='VarianceScaling',
                 expert_kernel_regularizer=None,
                 gate_kernel_regularizer=None,
                 expert_kernel_constraint=None,
                 gate_kernel_constraint=None,
                 activity_regularizer=None,
                 **kwargs):
        """
         Method for instantiating MMoE layer.

        :param units: 专家网络的隐藏单元的数量
        :param num_experts: 专家的数量
        :param num_tasks: 任务的数量
        :param use_expert_bias: 专家网络是否使用 bias
        :param use_gate_bias: 门控网络是否使用 bias
        :param expert_activation: 专家网络的激活函数
        :param gate_activation: 门控网络的激活函数
        :param expert_bias_initializer: Initializer for the expert bias
        :param gate_bias_initializer: Initializer for the gate bias
        :param expert_bias_regularizer: Regularizer for the expert bias
        :param gate_bias_regularizer: Regularizer for the gate bias
        :param expert_bias_constraint: Constraint for the expert bias
        :param gate_bias_constraint: Constraint for the gate bias
        :param expert_kernel_initializer: Initializer for the expert weights
        :param gate_kernel_initializer: Initializer for the gate weights
        :param expert_kernel_regularizer: Regularizer for the expert weights
        :param gate_kernel_regularizer: Regularizer for the gate weights
        :param expert_kernel_constraint: Constraint for the expert weights
        :param gate_kernel_constraint: Constraint for the gate weights
        :param activity_regularizer: Regularizer for the activity
        :param kwargs: Additional keyword arguments for the Layer class
        """
        # Hidden nodes parameter
        self.units = units
        self.num_experts = num_experts
        self.num_tasks = num_tasks

        # Weight parameter
        self.expert_kernels = None
        self.gate_kernels = None
        self.expert_kernel_initializer = initializers.get(expert_kernel_initializer)
        self.gate_kernel_initializer = initializers.get(gate_kernel_initializer)
        self.expert_kernel_regularizer = regularizers.get(expert_kernel_regularizer)
        self.gate_kernel_regularizer = regularizers.get(gate_kernel_regularizer)
        self.expert_kernel_constraint = constraints.get(expert_kernel_constraint)
        self.gate_kernel_constraint = constraints.get(gate_kernel_constraint)

        # Activation parameter
        self.expert_activation = activations.get(expert_activation)
        self.gate_activation = activations.get(gate_activation)

        # Bias parameter
        self.expert_bias = None
        self.gate_bias = None
        self.use_expert_bias = use_expert_bias
        self.use_gate_bias = use_gate_bias
        self.expert_bias_initializer = initializers.get(expert_bias_initializer)
        self.gate_bias_initializer = initializers.get(gate_bias_initializer)
        self.expert_bias_regularizer = regularizers.get(expert_bias_regularizer)
        self.gate_bias_regularizer = regularizers.get(gate_bias_regularizer)
        self.expert_bias_constraint = constraints.get(expert_bias_constraint)
        self.gate_bias_constraint = constraints.get(gate_bias_constraint)

        # Activity parameter
        self.activity_regularizer = regularizers.get(activity_regularizer)

        # Keras parameter
        self.input_spec = InputSpec(min_ndim=2)  # 至少两个维度
        self.supports_masking = True

        super(MMoE, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Method for creating the layer weights.

        :param input_shape: Keras tensor (future input to layer)
                            or list/tuple of Keras tensors to reference
                            for weight shape computations
        """
        assert input_shape is not None and len(input_shape) >= 2

        input_dimension = input_shape[-1]

        # 专家网络的参数，每个专家：(输入维度 * 每个专家的unit数量)
        self.expert_kernels = self.add_weight(
            name='expert_kernel',
            shape=(input_dimension, self.units, self.num_experts),
            initializer=self.expert_kernel_initializer,
            regularizer=self.expert_kernel_regularizer,
            constraint=self.expert_kernel_constraint,
        )

        # 每个专家的偏置bias：每个专家：(units, )
        if self.use_expert_bias:
            self.expert_bias = self.add_weight(
                name='expert_bias',
                shape=(self.units, self.num_experts),
                initializer=self.expert_bias_initializer,
                regularizer=self.expert_bias_regularizer,
                constraint=self.expert_bias_constraint,
            )

        # Initialize gate weights (number of input features * number of experts * number of tasks)
        self.gate_kernels = [self.add_weight(
            name='gate_kernel_task_{}'.format(i),
            shape=(input_dimension, self.num_experts),
            initializer=self.gate_kernel_initializer,
            regularizer=self.gate_kernel_regularizer,
            constraint=self.gate_kernel_constraint
        ) for i in range(self.num_tasks)]

        # Initialize gate bias (number of experts * number of tasks)
        if self.use_gate_bias:
            self.gate_bias = [self.add_weight(
                name='gate_bias_task_{}'.format(i),
                shape=(self.num_experts,),
                initializer=self.gate_bias_initializer,
                regularizer=self.gate_bias_regularizer,
                constraint=self.gate_bias_constraint
            ) for i in range(self.num_tasks)]

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dimension})

        super(MMoE, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """
        Method for the forward function of the layer.

        :param inputs: Input tensor
        :param kwargs: Additional keyword arguments for the base method
        :return: A tensor
        """
        gate_outputs = []
        final_outputs = []

        # f_{i}(x) = activation(W_{i} * x + b), 论文中激活函数是ReLU
        expert_outputs = tf.tensordot(a=inputs, b=self.expert_kernels, axes=1)  # (none, 4, 8)
        # 添加偏置，先将偏置调整成可以相加的形状
        if self.use_expert_bias:
            expert_bias_reshaped = tf.reshape(self.expert_bias, (1, self.units, self.num_experts))
            expert_outputs += expert_bias_reshaped  # 广播机制，(none, 4, 8) 与 偏置 (1, 4, 8)相加，右对齐后，扩展
        expert_outputs = self.expert_activation(expert_outputs)  # (none, 4, 8) 8个专家每个输出(none, 4)

        # g^{k}(x) = activation(W_{gk} * x + b), 激活函数softmax
        for index, gate_kernel in enumerate(self.gate_kernels):
            gate_output = keras.ops.dot(x1=inputs,
                                        x2=gate_kernel)  # (none, feature_dim) * (feature_dim, 8) 每个门控输出8个值，控制8个专家的权重
            if self.use_gate_bias:
                gate_output = tf.nn.bias_add(value=gate_output,
                                             bias=self.gate_bias[index])  # (none, 8) + (8,) 这个会广播机制进行相加
            gate_output = self.gate_activation(gate_output)  # (none, 8)
            gate_outputs.append(gate_output)

        # f^{k}(x) = sum_{i=1}^{n}(g^{k}(x)_{i} * f_{i}(x))
        for gate_output in gate_outputs:
            # 8个门控(none, 8), 构造成 4 组权重，然后元素级乘法后再相加
            expanded_gate_output = tf.expand_dims(gate_output, axis=1)  # (none, 1, 8)
            expanded_gate_output = tf.repeat(expanded_gate_output, self.units, axis=1)  # (none, 4, 8)
            weighted_expert_output = expert_outputs * expanded_gate_output  # (none, 4, 8) * (none, 4, 8) 元素级乘法，(none, 4, 8)
            final_outputs.append(
                tf.reduce_sum(weighted_expert_output, axis=-1))  # (none, 4) 最后一个维度求和，这样就得到了8个专家网络最后的输出的和
        return final_outputs  # [(none, 4), (none, 4)...]

    def compute_output_shape(self, input_shape):
        """
        Method for computing the output shape of the MMoE layer.

        :param input_shape: Shape tuple (tuple of integers)
        :return: List of input shape tuple where the size of the list is equal to the number of tasks
        """
        assert input_shape is not None and len(input_shape) >= 2

        output_shape = list(input_shape)
        output_shape[-1] = self.units
        output_shape = tuple(output_shape)

        return [output_shape for _ in range(self.num_tasks)]

    def get_config(self):
        """
        Method for returning the configuration of the MMoE layer.

        :return: Config dictionary
        """
        config = {
            'units': self.units,
            'num_experts': self.num_experts,
            'num_tasks': self.num_tasks,
            'use_expert_bias': self.use_expert_bias,
            'use_gate_bias': self.use_gate_bias,
            'expert_activation': activations.serialize(self.expert_activation),
            'gate_activation': activations.serialize(self.gate_activation),
            'expert_bias_initializer': initializers.serialize(self.expert_bias_initializer),
            'gate_bias_initializer': initializers.serialize(self.gate_bias_initializer),
            'expert_bias_regularizer': regularizers.serialize(self.expert_bias_regularizer),
            'gate_bias_regularizer': regularizers.serialize(self.gate_bias_regularizer),
            'expert_bias_constraint': constraints.serialize(self.expert_bias_constraint),
            'gate_bias_constraint': constraints.serialize(self.gate_bias_constraint),
            'expert_kernel_initializer': initializers.serialize(self.expert_kernel_initializer),
            'gate_kernel_initializer': initializers.serialize(self.gate_kernel_initializer),
            'expert_kernel_regularizer': regularizers.serialize(self.expert_kernel_regularizer),
            'gate_kernel_regularizer': regularizers.serialize(self.gate_kernel_regularizer),
            'expert_kernel_constraint': constraints.serialize(self.expert_kernel_constraint),
            'gate_kernel_constraint': constraints.serialize(self.gate_kernel_constraint),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer)
        }
        base_config = super(MMoE, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


mmoe_layer = MMoE(
    units=128,
    num_experts=8,
    num_tasks=1
)
# tensor_dict = fr.get_basic_feature_representation_case()
# feature_tensor = layers.concatenate([tensor_dict[k] for k in tensor_dict])
# mmoe_output = mmoe_layer(feature_tensor)
#
# fully_output = nn.FullyConnectedTower([64, 32, 1], 'test_tower', 'relu', 'sigmoid')(feature_tensor)
# print(fully_output)

# 获取特征的表示
inputs, tensor_dict = fr.get_basic_feature_representation()
feature_tensor = layers.concatenate([tensor_dict[k] for k in tensor_dict])
# MMoE层
mmoe_output = mmoe_layer(feature_tensor)
task_output = nn.FullyConnectedTower([64, 32, 1], 'ctr', 'relu', 'sigmoid')(mmoe_output[0])
# 模型
model = keras.models.Model(inputs=inputs, outputs=[task_output])
model.summary()
# 编译
model.compile(optimizer=keras.optimizers.Adam(0.0003),
              loss="binary_crossentropy",
              metrics=["AUC"])

logdir = "/home/web/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

# 获取dataset
dataset, test_dataset = data_process.train_test_dataset(1024)
# 训练数据
es = keras.callbacks.EarlyStopping(monitor='val_CTR_auc', patience=1, mode="max", restore_best_weights=True)
history = model.fit(dataset, epochs=3, validation_data=test_dataset, callbacks=[es, tensorboard_callback])

import tensorflow as tf
import keras.src.ops

import feature_representation.feature_representation as fr
from keras import layers, InputSpec
from keras import activations, initializers, regularizers, constraints, optimizers

import utils.nn_utils as nn
import datetime
from data_process import dataset_process

if __name__ == '__main__':
    inputs, tensor_dict = fr.get_basic_feature_representation()
    feature_tensor = layers.concatenate([tensor_dict[k] for k in tensor_dict])

    ctr_output = nn.FullyConnectedTower([512, 256, 128, 64, 1], 'ctr', 'relu', 'sigmoid')(feature_tensor)
    cvr_output = nn.FullyConnectedTower([512, 256, 128, 64, 1], 'cvr', 'relu', 'sigmoid')(feature_tensor)

    # 张量元素相乘
    ctcvr_output = keras.layers.Multiply(name='CTCVR')([ctr_output, cvr_output])
    # 模型
    model = keras.models.Model(inputs=inputs, outputs=[ctr_output, ctcvr_output])
    model.summary()
    # 编译
    model.compile(optimizer=keras.optimizers.Adam(0.0003),
                  loss=["binary_crossentropy", "binary_crossentropy"],
                  metrics=["AUC", "AUC"])

    logdir = "/home/web/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    # 获取dataset
    dataset, test_dataset = dataset_process.train_test_dataset(1024)
    # 训练数据
    es = keras.callbacks.EarlyStopping(monitor='val_ctr_auc', patience=1, mode="max", restore_best_weights=True)
    time1 = datetime.datetime.now()
    print(time1)
    history = model.fit(dataset, epochs=3, validation_data=test_dataset, callbacks=[es, tensorboard_callback])
    time2 = datetime.datetime.now()
    print(time2)
    time_interval = time2 - time1
    print("Training took:", time_interval)

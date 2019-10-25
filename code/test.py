import tensorflow as tf
import numpy as np
import pandas as pd
# 在vscode中tensorflow.keras没法自动补齐，tensorflow.python.keras可以，但是经常会出问题
# 在浏览器中tensorflow.keras能正常补齐
from tensorflow.keras import layers

def model0():
    model = tf.keras.Sequential()
    model.add(layers.Dense(64, activation = 'relu'))
    model.add(layers.Dense(64, activation = 'relu'))
    model.add(layers.Dense(10, activation = 'softmax'))

    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    data = np.random.random((1000, 32))
    labels = np.random.random((1000, 10))

    # model.fit(data, labels, epochs=10, batch_size=32)

    # val_data = np.random.random((100, 32))
    # val_labels = np.random.random((100, 10))
    # model.fit(data, labels, epochs=10, batch_size=32, validation_data=(val_data, val_labels))

    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.batch(32)
    dataset = dataset.repeat()
    model.fit(dataset, epochs=10, steps_per_epoch=30)

    # data1 = np.random.random((1000, 32))
    # labels1 = np.random.random((1000, 10))
    # model.evaluate(data1, labels1, batch_size=32)

def model1():
    inputs = tf.keras.Input(shape=(32, ))
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    predictions = layers.Dense(10, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    data = np.random.random((1000, 32))
    labels = np.random.random((1000, 10))
    model.fit(data, labels, batch_size=32, epochs=5)

class MyModel(tf.keras.Model):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__(name='my_model')
        self.num_classes = num_classes
        self.dense_1 = layers.Dense(32, activation='relu')
        self.dense_2 = layers.Dense(num_classes, activation='sigmoid')

    def call(self, inputs):
        x = self.dense_1(inputs)
        return self.dense_2(x)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)

def model2():
    model = MyModel(num_classes=10)
    model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    data = np.random.random((1000, 32))
    labels = np.random.random((1000, 10))
    model.fit(data, labels, batch_size=32, epochs=5)

model0()

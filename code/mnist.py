import tensorflow as tf
import numpy as np
import pandas as pd
# 在vscode中tensorflow.keras没法自动补齐，tensorflow.python.keras可以，但是经常会出问题
# 在浏览器中tensorflow.keras能正常补齐
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
x_train, x_test = x_train / 255.0, x_test / 255.0
np.shape(x_train), np.shape(y_train), np.shape(x_test), np.shape(y_test)

x_train_1 = np.reshape(x_train, (len(x_train), -1))
x_test_1 = np.reshape(x_test, (len(x_test), -1))
y_train_1 = pd.get_dummies(y_train)
y_test_1 = pd.get_dummies(y_test)

def model0():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test, verbose=2)

def model1():
    input = tf.keras.Input(shape=(28, 28,))
    o1 = layers.Flatten(input_shape=(28, 28))(input)
    o2 = layers.Dense(128, activation='relu')(o1)
    o3 = layers.Dropout(0.2)(o2)
    o4 = layers.Dense(10, activation='softmax')(o3)
    model = tf.keras.Model(inputs=input, outputs=o4)
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test, verbose=2)

def model2():
    input = tf.keras.Input(shape=(784,))
    o2 = layers.Dense(128, activation='relu')(input)
    o3 = layers.Dropout(0.2)(o2)
    o4 = layers.Dense(10, activation='softmax')(o3)
    model = tf.keras.Model(inputs=input, outputs=o4)
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    model.fit(x_train_1, y_train, epochs=5)
    model.evaluate(x_test_1, y_test, verbose=2)

model0()

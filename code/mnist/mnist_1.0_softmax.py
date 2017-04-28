# encoding: UTF-8
# Copyright 2016 Google.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import tensorflow as tf
import tensorflowvisu
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

tf.set_random_seed(0)

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
# reshape为false时返回的是[28, 28, 1]的数据，否则返784维向量
mnist = read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 10])
# flatten the images into a single line of pixels
# -1 in the shape definition means "the only possible dimension that will preserve the number of elements"
# -1表示根据已有数据，让程序算出来，只会有一种可能
XX = tf.reshape(X, [-1, 784])

# weights W[784, 10]   784=28*28
W = tf.Variable(tf.zeros([784, 10]))
# biases b[10]
b = tf.Variable(tf.zeros([10]))

# feed in 1 when testing, 0.75 when training
pkeep = tf.placeholder(tf.float32)

def one_layer_mode():
    # The model
    # tf.matmul是矩阵乘法
    return tf.matmul(XX, W) + b

# layers_dim: 如[784, 200, 100, 60, 30, 10]
def create_multi_layer(layers_dim):
    last_op = XX
    for i in range(len(layers_dim) - 1):
        input_dim = layers_dim[i]
        out_dim = layers_dim[i + 1]
        # 如果这里继续使用tf.zeros初始化，将准确率一直只有0.3，无法收敛
        # Wlocal = tf.Variable(tf.zeros([input_dim, out_dim]))
        # blocal = tf.Variable(tf.zeros([out_dim]))

        # truncated_normal：截断正态分布，stddev？
        Wlocal = tf.Variable(tf.truncated_normal([input_dim, out_dim], stddev = 0.1))
        # b还是用0来初始化?
        blocal = tf.Variable(tf.ones([out_dim]) / 10)
        last_op = tf.matmul(last_op, Wlocal) + blocal
        # 中间层的输出激活函数不用tf.nn.softmax
        if i < len(layers_dim) - 2:
            # sigmoid较难收敛，用relu代替
            # last_op = tf.nn.sigmoid(last_op)
            last_op = tf.nn.relu(last_op)
            # 可能过拟合，使用dropout每次随机禁用一些神经元
            # 同时等比例地促进剩余神经元的输出，以确保下一层的激活不会移动
            # 但是由于禁用的一些神经元，因此会导致准确度降低，导致出现准确率曲线不稳定
            # 训练时可以把pkeep设为小于1，测试时要pkeep = 1
            last_op = tf.nn.dropout(last_op, pkeep)

    return last_op

    # 如果返回Wlocal, blocal,会报以下错误
    # ValueError: Shapes must be equal rank, but are 2 and 1
    # From merging shape 1 with other shapes. for 'MatMul_1/a' (op: 'Pack') with input shapes: [?,200], [784,200], [200].
    # return last_op, Wlocal, blocal

# Ylogits = one_layer_mode()
# 层数越多，越难收敛，在0.5的学习率的情况下
# 3层+sigmoid ~= 准确度0.96, 3层+relu ~= 0.97
# Ylogits = create_multi_layer([784, 200, 100, 10])
# 5层+sigmoid ~= 准确度0.7664, 5层+relu ~= 0.97
Ylogits = create_multi_layer([784, 200, 100, 60, 30, 10])

Y = tf.nn.softmax(Ylogits)

# loss function: cross-entropy = - sum( Y_i * log(Yi) )
#                           Y: the computed output vector
#                           Y_: the desired output vector

# cross-entropy
# log takes the log of each element, * multiplies the tensors element by element
# reduce_mean will add all the components in the tensor
# so here we end up with the total cross-entropy for all images in the batch
# Y_ * tf.log(Y)是tf的张量乘积，就是把对应的元素相乘,必须相乘的对象shape是一样的
# reduce_mean是把tensor里面的元素全部累加求平均
# 原本的计算是* 1000， 是有问题的，会导致损失函数多乘以100，梯度相应的多乘以100，所以学习率必须降低100倍才能收敛
# 这是原本的学习率是0.005的原因，如果按照合理的乘以10，那学习率大概 > 2就不能收敛了。
# 学习率越大，收敛的越快，但是也可能导致震荡，无法收敛，一般0.1

# cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y_ * tf.log(Y), 1))这个计算是一样的
# tf.reduce_sum表示把tensor的所有元素求和，加上后面1参数（等价reduction_indices = [1]）表示对第二个维度求和
# shape(tf.reduce_sum(shape[100, 10])) = [100]
# normalized for batches of 100 images,
# *10 because  "mean" included an unwanted division by 10
# 使用softmax计算交叉熵
# 有可能出现Nan，因为试图计算log(0)，不应该使用这种方式
# 比如三层+relu就会出现Nan，导致最后准确度只有0.098
# cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y)) * 10.0

# softmax_cross_entropy_with_logits实现一般不会出现Nan
# 可能yi为0的话，不计算log(yi)，直接令yi_ * log(yi)为0
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = Ylogits, labels = Y_)
cross_entropy = tf.reduce_mean(cross_entropy)

# accuracy of the trained model, between 0 (worst) and 1 (best)
# tf.argmax(Y, 1)表示对Y的第2维求最大值，输出一个第一维数量的矩阵
# shape(tf.argmax(shape[a0, a1, a2, a3], 2)) = [a0, a1, a3]
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training, learning rate = 0.005
# 交叉熵计算乘以10，所以这里学习率应是0.5
learn_rate = tf.placeholder(tf.float32)
# GradientDescentOptimizer可能在鞍点处梯度为0，导致卡住，不能到达局部极小值点
# train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(cross_entropy)
# AdamOptimizer有一定的惯性，能越过鞍点，但是如果学习率较高，由于惯性可能导致震荡无法收敛
# 学习率相对GradientDescentOptimizer最后缩小100倍?
train_step = tf.train.AdamOptimizer(learn_rate).minimize(cross_entropy)

# matplotlib visualisation
# 0表示没有图像，1表示较少的图像，2表示较多的图像
vis_level = 0
count = 2001

if vis_level > 1:
    allweights = tf.reshape(W, [-1])
    allbiases = tf.reshape(b, [-1])
    I = tensorflowvisu.tf_format_mnist_images(X, Y, Y_)  # assembles 10x10 images by default
    It = tensorflowvisu.tf_format_mnist_images(X, Y, Y_, 1000, lines=25)  # 1000 images on 25 lines

if vis_level > 0:
    datavis = tensorflowvisu.MnistDataVis()

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

def cal_learn_rate(i):
    learn_rate_min = 0.0001
    learn_rate_max = 0.005
    return learn_rate_min + (learn_rate_max - learn_rate_min) * math.exp(- i / count)
    # return learn_rate_max

# You can call this function in a loop to train the model, 100 images at a time
def training_step(i, update_test_data, update_train_data):
    # training on batches of 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(100)
    train_feed_dict = {X: batch_X, Y_: batch_Y, learn_rate: cal_learn_rate(i), pkeep: 0.75}
    test_feed_dict = {X: mnist.test.images, Y_: mnist.test.labels, learn_rate: cal_learn_rate(i), pkeep: 1}

    # compute training values for visualisation
    if vis_level > 0:
        if update_train_data:
            a, c = sess.run([accuracy, cross_entropy], feed_dict = train_feed_dict)
            datavis.append_training_curves_data(i, a, c)
            if vis_level > 1:
                im, w, b = sess.run([I, allweights, allbiases], feed_dict = train_feed_dict)
                datavis.append_data_histograms(i, w, b)
                datavis.update_image1(im)
            print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c))

        # compute test values for visualisation
        if update_test_data:
            a, c = sess.run([accuracy, cross_entropy], feed_dict = test_feed_dict)
            datavis.append_test_curves_data(i, a, c)
            if vis_level > 1:
                im = sess.run(It, feed_dict = test_feed_dict)
                datavis.update_image2(im)
            print(str(i) + ": ********* epoch " + str(i*100//mnist.train.images.shape[0]+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))
    else:
        if update_train_data:
            a, c = sess.run([accuracy, cross_entropy], feed_dict = train_feed_dict)
            print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c))

        # compute test values for visualisation
        if update_test_data:
            a, c = sess.run([accuracy, cross_entropy], feed_dict = test_feed_dict)
            print(str(i) + ": ********* epoch " + str(i*100//mnist.train.images.shape[0]+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))

    # the backpropagation training step
    sess.run(train_step, feed_dict = train_feed_dict)

if vis_level > 0:
    datavis.animate(training_step, iterations=count, train_data_update_freq=10, test_data_update_freq=50, more_tests_at_start=True)
else:
    # to save the animation as a movie, add save_movie=True as an argument to datavis.animate
    # to disable the visualisation use the following line instead of the datavis.animate line
    # for i in range(count): training_step(i, False, False)
    for i in range(count): training_step(i, i % 50 == 0, i % 10 == 0)

if vis_level > 0:
    print("max test accuracy: " + str(datavis.get_max_test_accuracy()))
else:
    test_feed_dict = {X: mnist.test.images, Y_: mnist.test.labels, learn_rate: 0.01, pkeep: 1}
    print(sess.run(accuracy, feed_dict = test_feed_dict))

# final max test accuracy = 0.9268 (10K iterations). Accuracy should peak above 0.92 in the first 2000 iterations.

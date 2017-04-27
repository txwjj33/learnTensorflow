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

import tensorflow as tf
import tensorflowvisu
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
tf.set_random_seed(0)

# neural network with 1 layer of 10 softmax neurons
#
# · · · · · · · · · ·       (input data, flattened pixels)       X [batch, 784]        # 784 = 28 * 28
# \x/x\x/x\x/x\x/x\x/    -- fully connected layer (softmax)      W [784, 10]     b[10]
#   · · · · · · · ·                                              Y [batch, 10]

# The model is:
#
# Y = softmax( X * W + b)
#              X: matrix for 100 grayscale images of 28x28 pixels, flattened (there are 100 images in a mini-batch)
#              W: weight matrix with 784 lines and 10 columns
#              b: bias vector with 10 dimensions
#              +: add with broadcasting: adds the vector to each line of the matrix (numpy)
#              softmax(matrix) applies softmax on each line
#              softmax(line) applies an exp to each value then divides by the norm of the resulting line
#              Y: output matrix with 100 lines and 10 columns

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
# reshape为false时返回的是[28, 28, 1]的数据，否则返784维向量
mnist = read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 10])
# weights W[784, 10]   784=28*28
W = tf.Variable(tf.zeros([784, 10]))
# biases b[10]
b = tf.Variable(tf.zeros([10]))

# flatten the images into a single line of pixels
# -1 in the shape definition means "the only possible dimension that will preserve the number of elements"
# -1表示根据已有数据，让程序算出来，只会有一种可能
XX = tf.reshape(X, [-1, 784])

# The model
# tf.matmul是矩阵乘法
Y = tf.nn.softmax(tf.matmul(XX, W) + b)

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
cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y)) * 10.0  # normalized for batches of 100 images,
                                                          # *10 because  "mean" included an unwanted division by 10

# accuracy of the trained model, between 0 (worst) and 1 (best)
# tf.argmax(Y, 1)表示对Y的第2维求最大值，输出一个第一维数量的矩阵
# shape(tf.argmax(shape[a0, a1, a2, a3], 2)) = [a0, a1, a3]
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training, learning rate = 0.005
# 交叉熵计算乘以10，所以这里学习率应是0.5
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# matplotlib visualisation
use_vis = False
count = 1001

if use_vis:
    allweights = tf.reshape(W, [-1])
    allbiases = tf.reshape(b, [-1])
    I = tensorflowvisu.tf_format_mnist_images(X, Y, Y_)  # assembles 10x10 images by default
    It = tensorflowvisu.tf_format_mnist_images(X, Y, Y_, 1000, lines=25)  # 1000 images on 25 lines
    datavis = tensorflowvisu.MnistDataVis()

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# You can call this function in a loop to train the model, 100 images at a time
def training_step(i, update_test_data, update_train_data):
    # training on batches of 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(100)

    # compute training values for visualisation
    if use_vis and update_train_data:
        a, c, im, w, b = sess.run([accuracy, cross_entropy, I, allweights, allbiases], feed_dict={X: batch_X, Y_: batch_Y})
        datavis.append_training_curves_data(i, a, c)
        datavis.append_data_histograms(i, w, b)
        datavis.update_image1(im)
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c))

    # compute test values for visualisation
    if use_vis and update_test_data:
        a, c, im = sess.run([accuracy, cross_entropy, It], feed_dict={X: mnist.test.images, Y_: mnist.test.labels})
        datavis.append_test_curves_data(i, a, c)
        datavis.update_image2(im)
        print(str(i) + ": ********* epoch " + str(i*100//mnist.train.images.shape[0]+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))

    # the backpropagation training step
    sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y})


# datavis.animate(training_step, iterations=count, train_data_update_freq=10, test_data_update_freq=50, more_tests_at_start=True)

# to save the animation as a movie, add save_movie=True as an argument to datavis.animate
# to disable the visualisation use the following line instead of the datavis.animate line
for i in range(count): training_step(i, False, False)
# for i in range(count): training_step(i, i % 50 == 0, i % 10 == 0)

if use_vis:
    print("max test accuracy: " + str(datavis.get_max_test_accuracy()))
else:
    print(sess.run(accuracy, feed_dict={X: mnist.test.images, Y_: mnist.test.labels}))

# final max test accuracy = 0.9268 (10K iterations). Accuracy should peak above 0.92 in the first 2000 iterations.

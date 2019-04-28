#encoding: UTF-8
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import math

mnist = read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

W1 = tf.Variable(tf.truncated_normal([5, 5, 1, 4], stddev = 0.1))
B1 = tf.Variable(tf.ones([4]) / 10)
W2 = tf.Variable(tf.truncated_normal([5, 5, 4, 8],stddev = 0.1))
B2 = tf.Variable(tf.ones([8]) / 10)
W3 = tf.Variable(tf.truncated_normal([4, 4, 8, 12], stddev = 0.1))
B3 = tf.Variable(tf.ones([12]) / 10)

W4 = tf.Variable(tf.truncated_normal([7 * 7 * 12, 200], stddev = 0.1))
B4 = tf.Variable(tf.ones([200]) / 10)
W5 = tf.Variable(tf.truncated_normal([200, 10], stddev = 0.1))
B5 = tf.Variable(tf.ones([10]) / 10)

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y_ = tf.placeholder(tf.float32, [None, 10])
lr = tf.placeholder(tf.float32)

Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides = [1, 1, 1, 1], padding = "SAME") + B1)
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides = [1, 2, 2, 1], padding = "SAME") + B2)
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides = [1, 2, 2, 1], padding = "SAME") + B3)
Y4 = tf.reshape(Y3, [-1, 7 * 7 * 12])
Y5 = tf.nn.relu(tf.matmul(Y4, W4) + B4)
Ylogits = tf.matmul(Y5, W5) + B5
Y = tf.nn.softmax(Ylogits)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = Y_, logits = Ylogits)
cross_entropy = tf.reduce_mean(cross_entropy) * 100

correct_prediction = tf.equal(tf.argmax(Y_, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(3000):
    x_train, y_train = mnist.train.next_batch(100)

    min_lr = 0.0001
    max_lr = 0.003
    lr_i = min_lr + (max_lr - min_lr) * math.exp(- i / 2000)

    sess.run(train_step, {X: x_train, Y_: y_train, lr: lr_i})

    if i % 50 == 0:
        print("train accuracy: i:%d, %f" % (i, sess.run(accuracy, {X: x_train, Y_: y_train, lr: lr_i})))
        print("test accuracy: i: %d, %f" % (i, sess.run(accuracy, {X: mnist.test.images, Y_: mnist.test.labels})))

print("--------------------------------------------------------------")
print("test accuracy: %f" % sess.run(accuracy, {X: mnist.test.images, Y_: mnist.test.labels}))
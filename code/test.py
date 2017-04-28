#encoding: UTF-8

import tensorflow as tf

# a = [[[1, 1, 1], [2, 3, 4]], [[10, 10, 10], [2, 3, 4]], [[1, 1, 1], [2, 3, 4]]]
a1 = [[1, 1, 1], [2, 3, 4]]
a2 = [[1, 1, 1], [2, 3, 4]]
a3 = [1, 2, 3, 4]
a4 = [0.1, 0.2, 0.3, 0.9, 0.1, 0.43]

p1 = tf.placeholder(tf.float32, [None, 3])
p2 = tf.placeholder(tf.float32, [None, 3])
p3 = tf.placeholder(tf.float32)

c1 = tf.constant([[1, 1, 1], [2, 3, 4]])
c2 = tf.constant([[1, 1, 1], [2, 3, 4]])
c3 = tf.constant([10, 11, 12])
c4 = tf.constant([20, 21])

v1 = tf.Variable(tf.zeros([2, 3]))
v2 = tf.Variable(tf.ones([2, 3]))
v3 = tf.Variable(tf.ones(10) / 3)
v4 = tf.Variable(tf.truncated_normal([2, 3], stddev = 0.1))

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

def run(arg1, **args):
    print(sess.run(arg1, **args))

def test_plus():
    # run(a1 + a2) #不能运行
    # run(c1 + c2) #[[2 2 2], [4 6 8]]
    run(c1 + c3)  #[[11 12 13], [12 14 16]]
    # run(c1 + c4) #Dimensions must be equal, but are 3 and 2 for 'add_1' (op: 'Add') with input shapes: [2,3], [2].

def test_multi():
    # run(a1 * a2)    #不能运行
    # run(p1 * p2, feed_dict = {a1: [[1, 1, 1], [2, 3, 4]], b1: [[1, 1, 1], [2, 3, 4]]})  # 可以运行
    run(c1 * c2)  #可以运行

def test_reduce_sum_and_mean():
    run(tf.reduce_sum(a1, 1)) # [3, 9]
    # run(tf.reduce_sum(a1, reduction_indices = [1]))   # [3, 9]

    # run(tf.reduce_mean(a1))   #2

def test_argmax():
    # run(tf.argmax(a4, 1))  # 不能运行
    run(tf.argmax(a4, 0))  #3
    run(tf.argmax(a1, 0))  #[1, 1, 1]
    run(tf.argmax(a1, 1))  #[0, 2]

def test_init():
    run(v1)
    run(v2)
    run(v3)
    run(v4)

if __name__ == '__main__':
    test_init()



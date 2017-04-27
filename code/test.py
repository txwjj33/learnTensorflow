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

c1 = tf.constant([[1, 1, 1, 1], [2, 3, 4, 5]])
c2 = tf.constant([[1, 1, 1], [2, 3, 4]])

sess = tf.InteractiveSession()

def run(arg1, **args):
    print(sess.run(arg1, **args))

def test_reduce_sum_and_mean():
    # print(sess.run(a1 * a2))    #不能运行
    # print(sess.run(p1 * p2, feed_dict = {a1: [[1, 1, 1], [2, 3, 4]], b1: [[1, 1, 1], [2, 3, 4]]}))  # 可以运行
    # print(sess.run(c1 * c2))  #可以运行

    print(sess.run(tf.reduce_sum(a1, 1))) # [3, 9]
    # print(sess.run(tf.reduce_sum(a1, reduction_indices = [1])))   # [3, 9]

    # print(sess.run(tf.reduce_mean(a1)))   #2

def test_argmax():
    # run(tf.argmax(a4, 1))  # 不能运行
    run(tf.argmax(a4, 0))  #3
    run(tf.argmax(a1, 0))  #[1, 1, 1]
    run(tf.argmax(a1, 1))  #[0, 2]

if __name__ == '__main__':
    test_argmax()

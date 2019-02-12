import tensorflow as tf
import numpy as np

y = tf.placeholder(dtype=tf.int32, shape=[None], name="foo2")
z = tf.placeholder(dtype=tf.int32, shape=[None,1], name="foo3")

val2 = [1]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print(sess.run(y, feed_dict = {y: val2}))  # Fails
    print(sess.run(z, feed_dict = {z: tf.reshape(val2, []}))
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import os

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf

FLAGS = None
LOGDIR = 'C:/Users/lalo/Desktop/CCTVal/DLearningTF/tensorflow/mnist_example/checkpoints3/'

def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def deepnn(x):
    """deepnn builds the graph for a deep net for classifying digits.

    Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.

    Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
    """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob


def main(_):

    tf.reset_default_graph()
    sess = tf.Session()

    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Build the graph for the deep net
    y_conv, keep_prob = deepnn(x)



    with tf.name_scope('xent'):
        xent = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
        xent = tf.reduce_mean(xent)
        tf.summary.scalar("xent", xent)


    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(xent)

    # with tf.name_scope('accuracy'):
    #     correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    #     correct_prediction = tf.cast(correct_prediction, tf.float32)
    #     accuracy = tf.reduce_mean(correct_prediction, tf.float32)
    #     tf.summary.scalar("acc",accuracy)


    with tf.name_scope("accuracy"):
        # Returns the index with the largest value across axes of a tensor.
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        # Casts a tensor to a new type.
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    summ = tf.summary.merge_all()

    # saver = tf.train.Saver()
    # sess.run(tf.global_variables_initializer())
    # filewriter is how we write the summary protocol buffers to disk
    # writer = tf.summary.FileWriter(LOGDIR)
    # writer.add_graph(sess.graph)
    ## Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
    # config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    # tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)


    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    print('Saving graph to: %s' % LOGDIR)
    train_writer = tf.summary.FileWriter(LOGDIR)
    train_writer.add_graph(tf.get_default_graph())
    config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(train_writer, config)

    # with tf.Session() as sess:
    # sess.run(tf.global_variables_initializer())
    for i in range(100):
        batch = mnist.train.next_batch(10)
        if i % 10 == 0:
            print ('iteración %g: ' % i)
            """Evaluando accuracy y xent"""
            # train_accuracy = accuracy.eval(feed_dict={
            # x: batch[0], y_: batch[1], keep_prob: 1.0})
            # train_loss = xent.eval(feed_dict={
            # x: batch[0], y_: batch[1], keep_prob: 1.0})
            # print('step %d, training accuracy %g' % (i, train_accuracy))
            # print('step %d, training loss %g' % (i, train_loss))

            # tf.Summary(value=[tf.Summary.Value(tag='train_accuracy',
            #                                     simple_value=train_accuracy)])
            # tf.Summary(value=[tf.Summary.Value(tag='train_loss',
            #                                           simple_value=train_loss)])

            """tratando de loggear summaries con session"""
            [train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x: batch[0], y_: batch[1]})
            train_writer.add_summary(s, i)

            # summ = tf.summary.merge_all()

            # train_writer.add_summary(summ, i)




        if i % 50 == 0:
            print('hola')
            # saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), i)

        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})


    print('\ntest accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

    test_cross = correct_prediction.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})


    print('\nshape of cross entropy vector: %g' % np.shape(test_cross))
    print('accuracy form cross_entropy: %g' % (np.sum(test_cross)/np.shape(test_cross)))











if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

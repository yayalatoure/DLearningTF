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
LOGDIR = 'C:/Users/lalo/Desktop/CCTVal/checkpoints3/'

def conv2d(x, w):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv_layer(input, depth_in, depth_out,depth, name="conv"):

    with tf.name_scope(name):
        w = weight_variable([5, 5, depth_in, depth_out])
        b = bias_variable([depth_out])
        act = tf.nn.relu(conv2d(input, w) + b)
        '''Logging Histograms of Conv Layer'''
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)

        return act

def pool_layer(input, ds_h, ds_v, name="pool"):
    return tf.nn.max_pool(input, ksize=[1, ds_h, ds_v, 1],
                    strides=[1, ds_h, ds_v, 1], padding='SAME')




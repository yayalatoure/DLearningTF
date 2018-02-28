# Copyright 2017 Google, Inc. All Rights Reserved.

import os
import tensorflow as tf
import sys
import urllib

#versioning, urllib named differently for dif python versions
if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve

#define our github URLs
LOGDIR = '/temp/pico/'
GITHUB_URL ='https://raw.githubusercontent.com/mamcgrath/TensorBoard-TF-Dev-Summit-Tutorial/master/'

mnist = tf.contrib.learn.datasets.mnist.read_data_sets(train_dir=LOGDIR + 'data', one_hot=True)

'''
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
'''

urlretrieve(GITHUB_URL + 'labels_1024.tsv', LOGDIR + 'labels_1024.tsv')
urlretrieve(GITHUB_URL + 'sprite_1024.png', LOGDIR + 'sprite_1024.png')

def conv_layer(input, size_in, size_out, name="conv"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
        act = tf.nn.relu(conv + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)

        return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def fc_layer(input, size_in, size_out, name="fc"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        #fully connected part
        act = tf.nn.relu(tf.matmul(input, w) + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act

# build our model
def mnist_model(learning_rate, use_two_conv, use_two_fc, hparam):

    tf.reset_default_graph()
    sess = tf.Session()

    x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    # Outputs a Summary protocol buffer with images.
    tf.summary.image('input', x_image, 3)
    # for the labels
    y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")

    # 2 conv layers or 1?
    if use_two_conv:
        conv1 = conv_layer(x_image, 1, 32, "conv1")
        conv_out = conv_layer(conv1, 32, 64, "conv2")
    else:
        conv1 = conv_layer(x_image, 1, 64, "conv")
        conv_out = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    flattened = tf.reshape(conv_out, [-1, 7 * 7 * 64])

    # 2 fully connected layers or one?
    if use_two_fc:
        # give it the flattened image tensor
        fc1 = fc_layer(flattened, 7 * 7 * 64, 1024, "fc1")
        # we want these embeeddings to visualize them later
        embedding_input = fc1
        embedding_size = 1024
        logits = fc_layer(fc1, 1024, 10, "fc2")
    else:
        # else we take them directly from the conv layer
        embedding_input = flattened
        embedding_size = 7 * 7 * 64
        # logits the sum of the inputs may not equal 1, that the values are not probabilities
        # we'll feed these to the last (softmax) to make them probabilities
        logits = fc_layer(flattened, 7 * 7 * 64, 10, "fc")

    # short for cross entropy loss
    with tf.name_scope("xent"):
        # Computes the mean of elements across dimensions of a tensor.
        # so in this case across output probabilties
        xent = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=y), name="xent")
        # save that single number
        tf.summary.scalar("xent", xent)

    with tf.name_scope("train"):
        # Adam offers several advantages over the simple tf.train.GradientDescentOptimizer.
        # Foremost is that it uses moving averages of the parameters (momentum);
        # This enables Adam to use a larger effective step size, and the algorithm will converge to this step size without fine tuning.
        # The main down side of the algorithm is that Adam requires more computation to be performed for each parameter
        # in each training step (to maintain the moving averages and variance, and calculate the scaled gradient);
        # and more state to be retained for each parameter (approximately tripling the size of the model to store the average and variance for each parameter).
        # A simple tf.train.GradientDescentOptimizer could equally be used in your MLP, but would require more hyperparameter tuning before it would converge as quickly.
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(xent)

    with tf.name_scope("accuracy"):
        # Returns the index with the largest value across axes of a tensor.
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        # Casts a tensor to a new type.
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    # merge them all so one write to disk, more comp efficient
    summ = tf.summary.merge_all()

    # intiialize embedding matrix as 0s
    embedding = tf.Variable(tf.zeros([1024, embedding_size]), name="test_embedding")
    # give it calculated embedding
    assignment = embedding.assign(embedding_input)
    # initialize the saver
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    # filewriter is how we write the summary protocol buffers to disk
    writer = tf.summary.FileWriter(LOGDIR)
    writer.add_graph(sess.graph)

    ## Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
    config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    ## You can add multiple embeddings. Here we add only one.
    embedding_config = config.embeddings.add()
    embedding_config.tensor_name = embedding.name
    embedding_config.sprite.image_path = LOGDIR + 'sprite_1024.png'
    embedding_config.metadata_path = LOGDIR + 'labels_1024.tsv'
    # Specify the width and height of a single thumbnail.
    embedding_config.sprite.single_image_dim.extend([28, 28])
    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)


    # training step
    for i in range(2001):
        batch = mnist.train.next_batch(100)
        if i % 10 == 0:
            [train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x: batch[0], y: batch[1]})
            writer.add_summary(s, i)
        if i % 500 == 0:
            sess.run(assignment, feed_dict={x: mnist.test.images[:1024], y: mnist.test.labels[:1024]})
            # save checkpoints
            saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), i)
        sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})

def make_hparam_string(learning_rate, use_two_fc, use_two_conv):
    conv_param = "conv=2" if use_two_conv else "conv=1"
    fc_param = "fc=2" if use_two_fc else "fc=1"
    return "lr_%.0E,%s,%s" % (learning_rate, conv_param, fc_param)

def main():
# You can try adding some more learning rates
    for learning_rate in [1E-4]:
        # Include "False" as a value to try different model architectures
        for use_two_fc in [True]:
            for use_two_conv in [True]:
                # Construct a hyperparameter string for each one (example: "lr_1E-3,fc=2,conv=2)
                hparam = make_hparam_string(learning_rate, use_two_fc, use_two_conv)
                print('Starting run for %s' % hparam)

                # Actually run with the new settings
                mnist_model(learning_rate, use_two_fc, use_two_conv, hparam)


if __name__ == '__main__':
    main()

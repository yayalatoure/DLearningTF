from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot
import matplotlib as mpl

tf.logging.set_verbosity(tf.logging.INFO)

LOGDIR = 'C:/Users/lalo/Desktop/CCTVal/DLearningTF/tensorflow/mnist_example/checkpoints1-1/'

def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()

def cnn_model_fn(features, labels, mode):

    """Se incorpora save, para ver si se puede guardar
    accuracy y lost"""
    # tf.reset_default_graph()
    # sess = tf.Session()


    """Model function for CNN."""
    # Input Layer : [batch_size, image_width, image_height, channels]
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # tf.summary.image('input', input_layer, 3)

    # Convolutional Layer N°1
    # padding = same => se considera zero padding
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer N°1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer N°2 and Pooling Layer N°2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer N°1
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1, name="clases"),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    """Intentar hacer el logging de datos"""

    # summ = tf.summary.merge_all()
    # saver = tf.train.Saver()
    # sess.run(tf.global_variables_initializer())
    # writer = tf.summary.FileWriter(LOGDIR)
    # writer.add_graph(sess.graph)
    # saver.save(sess, os.path.join(LOGDIR, "model.ckpt"))
    #
    # ## Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
    # config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    #
    # tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)


    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    # writer.add_summary(loss)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    print('Size of training data is: ', np.shape(train_data))
    print('Size of training labels is: ', np.shape(train_labels))
    print('Size of evaluation data is: ', np.shape(eval_data))
    print('Size of evaluation data is: ', np.shape(eval_labels))

    image = np.reshape(train_data[0], (28, 28))
    print('Size of image is: ', np.shape(image))
    show(image)

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="C:/Users/lalo/Desktop/CCTVal/DLearningTF/tensorflow/mnist_example/checkpoints1")
    # Directorio anterior para los checkpoints de datos del modelo serán guardados.

    # Set up logging for predictions
    tensors_to_log = {"probabilities":"softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=2)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=10,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=200,
        hooks=[logging_hook])

    # Evaluate the model and print results
    # La evaluacion se raliza con la m+etrica definida
    # anteriormente en: cnn_model_fn -> eval_metric_ops
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

if __name__ == "__main__":
    tf.app.run()

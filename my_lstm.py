# coding=utf-8
'''
A Reccurent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import pdb
import reader
import time
import sys

# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("tmp/data/recurrent_network", one_hot=True)
# raw_data = reader.bp_raw_data_batch('dataset/makeup')
raw_data = reader.bp_raw_data_batch('/home/zhangkaihao/software/workspace/lxh')
train_data, train_data_len, test_data, test_data_len = raw_data
#pdb.set_trace()
'''
To classify images using a reccurent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''
# Parameters
# learning_rate = 0.001
learning_rate = tf.Variable(0.001, trainable=False)
training_iters = 200
batch_size = 125
display_step = 10

# Network Parameters
n_input = 17 # IHealth data input
n_profile = 4 # user profile input
n_steps = 18 # timesteps
n_hidden = 100 # hidden layer num of features
n_reluhidden = 100 # hidden layer num of relu nn

# tf Graph input
# 输入数据
x = tf.placeholder("float", [batch_size, n_steps, n_input])
x_profile = tf.placeholder("float", [batch_size, n_profile])
# 目标数据
y = tf.placeholder("float", [batch_size, n_steps, 1])
# 在变长LSTM中，表示每一个输入序列数据的步长n_steps
w = tf.placeholder(tf.int32, [batch_size, 1])
# 在变长LSTM中，以向量的方式表示每个序列数据的步长n_steps，例如某个序列数据的n_steps是2，则用【0,1.0,0,0,...】，主要用来计算测试集合的MAE
z = tf.placeholder(tf.int32, [batch_size, n_steps, 1])

# Define weights
weights = {
    'reluhidden_out': tf.Variable(tf.random_normal([n_reluhidden, 1])),
    'reluhidden_in': tf.Variable(tf.truncated_normal([n_hidden, n_reluhidden], stddev=0.0001)),
    'reluhidden_in_biases': tf.Variable(tf.ones([n_reluhidden])),
    'profile_out': tf.Variable(tf.random_normal([n_profile, n_reluhidden]))
}
'''
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}
'''


def RNN(x, weights):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    # pdb.set_trace()
    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, n_steps, x)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Define a multi layers lstm cell: multi_lstm_cell
    lstm_cell = rnn_cell.MultiRNNCell([lstm_cell] * 2)

    #pdb.set_trace()
    # Get lstm cell output
    # https://github.com/tensorflow/tensorflow/blob/r0.8/tensorflow/python/ops/rnn.py
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)
    #sequence_length参数暂时不会用，可以先不用。只不过计算速度可能会稍微慢一些。 sequence_length.shape=batch_size*n_hidden
    # outputs, states = rnn.rnn(lstm_cell, x, sequence_length=w, dtype=tf.float32)

    # Linear activation, using rnn inner loop every output
    pred_list = []
    for output_step in outputs:
        reluinput = tf.add(tf.matmul(x_profile, weights['profile_out']), output_step)
        hidden_layer_1 = tf.nn.relu(tf.matmul(reluinput, weights['reluhidden_in']) + weights['reluhidden_in_biases'])   # Question + 的执行过程
        pred_list.append(tf.matmul(hidden_layer_1, weights['reluhidden_out']))
    # return tf.matmul(outputs[-1], weights['out']), outputs, states
    return pred_list

#pdb.set_trace()
#pred, outputs, states = RNN(x, weights)
pred_list = RNN(x, weights)
pred = tf.pack(pred_list)
pred = tf.transpose(pred, [1, 0, 2])


# Define loss and optimizer
# method one
'''
pred_flag = tf.ceil(y)
cost = 0
for batch_index in range(batch_size):
    cost += tf.div(tf.reduce_sum(tf.mul(tf.div(tf.square(tf.sub(pred[batch_index,:,:], y[batch_index,:,:])), tf.constant(2, tf.float32, shape=[1,])), pred_flag[batch_index,:,:])), tf.reduce_sum(pred_flag[batch_index,:,:]))
#pdb.set_trace()
cost = cost / batch_size
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
'''
# method 2
pred_rmse = tf.reshape(pred, [-1,1])
y_rmse = tf.reshape(y, [-1,1])
pred_rmse_flag = tf.ceil(y_rmse)
z_rmse = tf.reshape(z, [-1,1])
# (pred-y)^2/2
cost = tf.div(tf.square(tf.sub(pred_rmse, y_rmse)), tf.constant(2, tf.float32, shape=[1,]))
cost = tf.div(tf.reduce_sum(tf.mul(cost, pred_rmse_flag)), tf.reduce_sum(pred_rmse_flag))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model on test dataset
# correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# method one
'''
mae = 0
for batch_index in range(batch_size):
    mae += tf.reduce_sum(tf.mul(tf.abs(tf.sub(pred[batch_index,:,:], y[batch_index,:,:])), tf.to_float(z[batch_index,:,:])))
mae = mae / batch_size
'''
# method two
# Reshape pred(125,14,,1)
pred_mae = tf.reshape(pred, [-1,1])
y_mae = tf.reshape(y, [-1,1])
z_mae = tf.reshape(z, [-1,1])
mae = tf.div(tf.reduce_sum(tf.abs(tf.mul(tf.sub(pred_mae, y_mae), tf.to_float(z_mae)))), tf.to_float(tf.reduce_sum(z_mae)))
rmse_pred = tf.div(tf.reduce_sum(tf.square(tf.mul(tf.sub(pred_mae, y_mae), tf.to_float(z_mae)))), tf.to_float(tf.reduce_sum(z_mae)))


# Initializing the variables
init = tf.initialize_all_variables()

# Define epoch function
def run_epoch(session, data, datelen, learning_rate_variable, eval_op, verbose=False):
    """Runs the model on the given data."""
    #pdb.set_trace()
    epoch_size = len(data) // batch_size
    start_time = time.time()
    results = 0
    for step, (batch_x, batch_x_profile, batch_y, batch_z) in enumerate(reader.bp_iterator_batch_secondary(data, datelen, batch_size, n_steps)):
        result, _ = session.run([cost, eval_op], feed_dict={x: batch_x, x_profile: batch_x_profile, y: batch_y, learning_rate: learning_rate_variable})
        results += result
    return results / (step+1)

# Define epoch function
def run_epoch_test(session, data, datelen, learning_rate_variable, eval_op, verbose=False):
    """Runs the model on the given data."""
    # pdb.set_trace()
    epoch_size = len(data) // batch_size
    start_time = time.time()
    results = 0
    rmse_pred_results = 0
    for step, (batch_x, batch_x_profile ,batch_y, batch_z) in enumerate(reader.bp_iterator_batch_secondary(data, datelen, batch_size, n_steps)):
        rmse_pred_result, result = session.run([rmse_pred, eval_op], feed_dict={x: batch_x, x_profile: batch_x_profile, y: batch_y, z: batch_z, learning_rate: learning_rate_variable})
        results += result
        rmse_pred_results += rmse_pred_result
    return results / (step+1), rmse_pred_results / (step+1)

# Launch the graph
# Assume that you have 12GB of GPU memory and want to allocate ~4GB:
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess, tf.device('/gpu:0'):
    sess.run(init)
    step = 0
    mae_pre = sys.float_info.max
    poscnt = 0 # Positive count
    # Keep training until reach max iterations
    learning_rate_variable = 0.001
    while step < training_iters:
        # batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        # batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        # Run optimization op (backprop)
        # sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if poscnt >= 5:
            poscnt = 0
            # learning_rate_variable = learning_rate_variable * 0.5
            #learning_rate_variable = 0.0001
        rmse_result = run_epoch(sess, train_data, train_data_len, learning_rate_variable, optimizer)
        rmse_pred_result, mae_result = run_epoch_test(sess, test_data, test_data_len, learning_rate_variable, mae)
        '''
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print "Iter " + str(step*batch_size) + ", Minibatch Loss= " +
                  "{:.6f}".format(loss) + ", Training Accuracy= " +
                  "{:.5f}".format(acc)
        '''
        if mae_result - mae_pre >= 0: # mae 变大
            poscnt += 1
        else:
            poscnt = 0
        mae_pre = mae_result

        step += 1
        print "   MAE:", "{:.13f}".format(mae_result), "  TEST RMSE:", "{:.13f}".format(rmse_pred_result), "Learning Rate:", learning_rate_variable, "Training Iters:", step
    print "Optimization Finished!"

    # Calculate accuracy for 128 mnist test images
    '''
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print "Testing Accuracy:",
        sess.run(accuracy, feed_dict={x: test_data, y: test_label})
    '''

'''
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    pdb.set_trace()
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print "Iter " + str(step*batch_size) + ", Minibatch Loss= " +
                  "{:.6f}".format(loss) + ", Training Accuracy= " +
                  "{:.5f}".format(acc)
        step += 1
    print "Optimization Finished!"

    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    #print "Testing Accuracy:",
sess.run(accuracy, feed_dict={x: test_data, y: test_label})
'''
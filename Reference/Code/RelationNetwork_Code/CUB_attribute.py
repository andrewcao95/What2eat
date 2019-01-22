import tensorflow as tf
import numpy as np, h5py
import scipy.io as sio
import sys
import random
import kNN
import re
import os
from numpy import *   


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def compute_accuracy(test_att, test_visual, test_id, test_label):
    global left_a2 
    att_pre = sess.run(left_a2, feed_dict={att_features: test_att})
    test_id = np.squeeze(np.asarray(test_id))
    outpre = [0]*2933 
    test_label = np.squeeze(np.asarray(test_label))
    test_label = test_label.astype("float32")
    for i in range(2933): 
        outputLabel = kNN.kNNClassify(test_visual[i,:], att_pre, test_id, 1) 
        outpre[i] = outputLabel
    correct_prediction = tf.equal(outpre, test_label)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={att_features: test_att, visual_features: test_visual})
    return result


f=sio.loadmat('./data/CUB_data/train_attr.mat')
att=np.array(f['train_attr'])
att.shape

f=sio.loadmat('./data/CUB_data/train_cub_googlenet_bn.mat')
x=np.array(f['train_cub_googlenet_bn'])
x.shape

f=sio.loadmat('./data/CUB_data/test_cub_googlenet_bn.mat')
x_test=np.array(f['test_cub_googlenet_bn'])
x_test.shape

f=sio.loadmat('./data/CUB_data/test_labels_cub.mat')
test_label=np.array(f['test_labels_cub'])
test_label.shape

f=sio.loadmat('./data/CUB_data/testclasses_id.mat')
test_id=np.array(f['testclasses_id'])

f=sio.loadmat('./data/CUB_data/test_proto.mat')
att_pro=np.array(f['test_proto'])


# # data shuffle
def data_iterator():
    """ A simple data iterator """
    batch_idx = 0
    while True:
        # shuffle labels and features
        idxs = np.arange(0, len(x))
        np.random.shuffle(idxs)
        shuf_visual = x[idxs]
        shuf_att = att[idxs]
        batch_size = 100
        for batch_idx in range(0, len(x), batch_size):
            visual_batch = shuf_visual[batch_idx:batch_idx+batch_size]
            visual_batch = visual_batch.astype("float32")
            att_batch = shuf_att[batch_idx:batch_idx+batch_size]
            yield att_batch, visual_batch
            



# # Placeholder
# define placeholder for inputs to network
att_features = tf.placeholder(tf.float32, [None, 312])
visual_features = tf.placeholder(tf.float32, [None, 1024])


# # Network

# CUB 312 700 1024 ReLu, 1e-2 * regularisers, 100 batch, 0.00001 Adam
W_left_a1 = weight_variable([312, 700])
b_left_a1 = bias_variable([700])
left_a1 = tf.nn.relu(tf.matmul(att_features, W_left_a1) + b_left_a1)

W_left_a2 = weight_variable([700, 1024])
b_left_a2 = bias_variable([1024])
left_a2 = tf.nn.relu(tf.matmul(left_a1, W_left_a2) + b_left_a2)


# # loss

loss_a = tf.reduce_mean(tf.square(left_a2 - visual_features))

# L2 regularisation for the fully connected parameters.
regularizers_a = (tf.nn.l2_loss(W_left_a1) + tf.nn.l2_loss(b_left_a1)
                + tf.nn.l2_loss(W_left_a2) + tf.nn.l2_loss(b_left_a2))
               

                  
# Add the regularization term to the loss.
loss_a += 1e-2 * regularizers_a



train_step = tf.train.AdamOptimizer(0.00001).minimize(loss_a)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


# # Run
iter_ = data_iterator()
for i in range(1000000):
    att_batch_val, visual_batch_val = iter_.next()
    sess.run(train_step, feed_dict={att_features: att_batch_val, visual_features: visual_batch_val})
    if i % 1000 == 0:
        print(compute_accuracy(att_pro, x_test, test_id, test_label))






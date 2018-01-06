import tensorflow as tf
import numpy as np
import os

# set the specific GPU
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

# Input
input_real = tf.placeholder(tf.float32, shape=(None, None), name='input_real') # here the shape is None, 28*28
input_fake = tf.placeholder(tf.float32, shape=(None, None), name='input_fake')

# check the mnist picture
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')

# generator
def generator(z, input_dimension=100, out_dimension=784, activation=relu):
    '''
    z: the noise vector, should be a tensor
    input_dimension: the size of z.
    out_dimension: the picture size.
    hidden_layers: hidden layer number.
    activation: relu default.
    '''
    tf.layers.dense(z, )

with tf.variable_scope('generator'):
    hidden_layer = tf.layers.dense(input_fake, 128, activation=tf.nn.tanh, name='hidden_layer')
    output = tf.layers.dense(hidden_layer, 784, name='output')

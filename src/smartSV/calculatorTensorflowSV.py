"""
project: Smart Village
CalculatorTensorflowSV librairy
Author: HO Van Hieu
"""

import numpy as np
import tensorflow as tf 
from tensorflow.contrib import rnn
import vocabulariesSV as vc



# convert a tensor to a numpy array
def tensorflow_to_npArray(tensor):
    sess = tf.Session()
    with sess.as_default():
        numpy_array = tensor.eval()
    return numpy_array

# convert a numpy array to a tensor
def npArray_to_tensorflow(np_array):
    sess = tf.Session()
    with sess.as_default():
        tensor = tf.constant(np_array)
        
    return tensor

# Get predict of Al RegressionLineair in Tensorflow
def getPredict_LinearRegression_Tensorflow(X, W, b):
        predict = np.add(np.matmul(X, W), b)
        return predict

# Get predict of Al Neron Network 2 hidden layers in Tensorflow
def getPredict_NN2hiddenlayers_Tensorflow(X, weights, biases):
        # Hidden fully connected layer with 18 neurons
        layer_1 = np.add(np.matmul(X, weights['h1']), biases['b1'])
        # Hidden fully connected layer with 18 neurons
        layer_2 = np.add(np.matmul(layer_1, weights['h2']), biases['b2'])
        # Output fully connected layer with a neuron for each class
        out_layer = np.matmul(layer_2, weights['out']) + biases['out']
        return out_layer

#Conv2D wrapper, with Weights and bias
def conv2d(x, W, b, stride=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

# MaxPool2D wrapper
def maxpool2d(x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    
def conv_net(x, weights, biases, dropout, nb_kernel):
        # data input is a 1-D vector of nX features 
        # Reshape to match the format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, daysnumber, Width, Channel]
        nX = (int) (x.shape[1])
        #print(nX)
        x = tf.reshape(x, shape=[-1,  nb_kernel, int (nX/nb_kernel), 1]) 
        
        # Convolution Layer
    
        conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    
        # Max Pooling (down-sampling)
        conv1 = maxpool2d(conv1, k=2)
    
        # Convolution Layer
        conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
        # Max Pooling (down-sampling)
        conv2 = maxpool2d(conv2, k=2)
    
        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].shape[0]])
        fc1 = tf.cast(fc1, tf.float32)
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
    
        # Apply Dropout
        fc1 = tf.nn.dropout(fc1, dropout)
        # Output, class prediction
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        return out    
    
# Get predict of Al CNN in Tensorflow
def getPredict_CNN_Tensorflow(X, weights, biases, nb_kernel):
    predict = conv_net(X, weights, biases, vc.keep_prob, nb_kernel)
    predict = tensorflow_to_npArray(predict)
    
    return predict

# Define the function RNNF to get predict of Al RNN in Tensorflow
def RNNF(x, weights, biases, timesteps):
        # Prepare data shape to match 'rnn' function requirements
        # Current data input shape: (batch_size, timesteps, nX)
        # Required shape: 'timesteps' tensors list of shape (batch_size, nY)

        # Unstack to get a list of 'timesteps' tensors of shape (batch_size, nX)
        nX = (int) (x.shape[1] / timesteps)
        x = npArray_to_tensorflow(x)
        x = tf.cast(x, tf.float32)
        #tf.reshape(x, [_batch_size, -1])
        x = tf.reshape(x, [-1, timesteps, nX])
        x = tf.unstack(x, timesteps, 1)

        # Define a lstm cell with tensorflow
        try:
            lstm_cell = rnn.BasicLSTMCell(vc.nbhidden, forget_bias=1.0, reuse= True  ) #
        except Exception as e:
            lstm_cell = rnn.BasicLSTMCell(vc.nbhidden, forget_bias=1.0 , reuse= False )
        
        
        # Linear activation, using rnn inner loop last output
        try:
            outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32 )
        except Exception as e:
            outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float64)

        
        predict = tf.add(tf.matmul(outputs[-1], weights['out']),biases['out'])
        
        return predict
    
# Get predict of Al RNN in Tensorflow
def getPredict_RNN_Tensorflow(x, weights, biases, timesteps):    
    
        init_op       = tf.global_variables_initializer()
        predict =  RNNF(x, weights, biases, timesteps)
        with tf.Session() as sess:
                sess.run(init_op) 
                array = sess.run(predict)
        
        return array
       
        
# Define the function BiRNNF to get predict of Al BiRNN in Tensorflow
def BiRNNF(x, weights, biases, timesteps):

        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, timesteps, n_input)
        # Required shape: 'timesteps' tensors list of shape (batch_size, num_input)

        # Unstack to get a list of 'timesteps' tensors of shape (batch_size, num_input)
        nX = (int) (x.shape[1] / timesteps)
        x = npArray_to_tensorflow(x)
        x = tf.cast(x, tf.float32)
        #tf.reshape(x, [_batch_size, -1])
        x = tf.reshape(x, [-1, timesteps, nX])
        x = tf.unstack(x, timesteps, 1)

        # Define lstm cells with tensorflow
        
        # Forward direction cell
        try:
            lstm_fw_cell = rnn.BasicLSTMCell(vc.nbhidden, forget_bias=1.0, reuse= True  ) #
        except Exception as e:
            lstm_fw_cell = rnn.BasicLSTMCell(vc.nbhidden, forget_bias=1.0 , reuse= False )
            
        # Backward direction cell
        try:
            lstm_bw_cell = rnn.BasicLSTMCell(vc.nbhidden, forget_bias=1.0, reuse= True  ) #
        except Exception as e:
            lstm_bw_cell = rnn.BasicLSTMCell(vc.nbhidden, forget_bias=1.0 , reuse= False )
        
        # Get lstm cell output
        try:
            outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                              dtype=tf.float32)
        except Exception: # Old TensorFlow version only returns outputs not states
            outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                        dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], weights['out']) + biases['out']

# Get predict of Al RNN in Tensorflow
def getPredict_BiRNN_Tensorflow(x, weights, biases, timesteps):    
        init_op       = tf.global_variables_initializer()
        predict =  BiRNNF(x, weights, biases, timesteps)
        with tf.Session() as sess:
                sess.run(init_op) 
                array = sess.run(predict)
        
        return array
'''
TensorFlowSV library.
Project: Smart Village
Prediction
Author:  HO Van Hieu
'''
from __future__ import print_function
import tensorflow as tf 
from tensorflow.contrib import rnn
import numpy as np
import calculatorSV as cc
import calculatorTensorflowSV as ccT
import vocabulariesSV as vc
import time

rng = np.random



#defini a class model
class model_tensorflow(object):
    #To build the model when instantiate
    # method: name of method (CNN, RNN ..?)
    # trainX: DataSet for training
    # trainY: LabelSet for training
    def __init__(self, method, Xtrain, Ytrain):
        self.method = method
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        
        #self.Xtest = Xtest
        #self.Ytest = Ytest
        
        
        self.nX = Xtrain.shape[1]
        self.nY = Ytrain.shape[1]
        self.nb_hd1 = self.nX + 5
        self.nb_hd2 = self.nY + 5
        self.nb_kernel =  cc.choose_divisor(self.nX)
        self.model_method = self.set_model_method()
        self.weights, self.biases = self.set_weights_biases()
    
    # to buil model (training phase)
    # trainX: DataSet for training - numpyarray
    # trainY: LabelSet for training - numpyarray
    def set_model_method(self):
        if( self.method == vc.LinearRegression):
             model_method = LinearRegression(self.nX, self.nY)
        elif (self.method == vc.NN2hd):
             model_method = NN2hiddenlayers(self.nX, self.nY,self.nb_hd1, self.nb_hd2)
        elif (self.method == vc.CNN):
             model_method= CNN(self.nX, self.nY, self.nb_kernel)
        elif (self.method == vc.RNN):
             model_method= RNN(self.nX, self.nY, self.nb_kernel)
        elif (self.method == vc.BiRNN):
             model_method= BiRNN(self.nX, self.nY, self.nb_kernel)
        else:
            model_method = None
        
        return model_method
    
    # calcul weights and biases
    def set_weights_biases(self):
        weights, biases = self.model_method.launchG(self.Xtrain, self.Ytrain)
        
        return weights, biases
        
    # get the prediction
    # Xtest: DataSet - numpyarray
    # return: the prediction - numpyarray
    def get_predict(self, Xtest):
        if( self.method == vc.LinearRegression):
            predict = ccT.getPredict_LinearRegression_Tensorflow(
                 Xtest, self.weights, self.biases)
        
        elif (self.method == vc.NN2hd):
            predict = ccT.getPredict_NN2hiddenlayers_Tensorflow(
                 Xtest, self.weights, self.biases)
        elif (self.method == vc.CNN):
            testX = ccT.npArray_to_tensorflow(Xtest)
            predict = ccT.getPredict_CNN_Tensorflow(
                 testX, self.weights, self.biases, self.nb_kernel)
        elif (self.method == vc.RNN):
            testX = ccT.npArray_to_tensorflow(Xtest)
            predict = ccT.getPredict_RNN_Tensorflow(Xtest, self.weights, self.biases, self.nb_kernel)
        elif (self.method == vc.BiRNN):
            testX = ccT.npArray_to_tensorflow(Xtest)
            predict = ccT.getPredict_BiRNN_Tensorflow(
                 Xtest, self.weights, self.biases, self.nb_kernel)
        else:
            predict = None
        
        return predict
        

#defini a class of graph linear Regression
class LinearRegression(object):

    #To build the graph when instantiated
    def __init__(self, nX, nY):
        self.learning_rate = vc.learning_rate
        self.num_steps = vc.nb_epochs
        self.batch_size = vc.batch_size
        self.dropout = vc.keep_prob
        self.keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)
        self.X = tf.placeholder("float", [None, nX])
        self.Y = tf.placeholder("float", [None, nY])
        self.sizeY = tf.placeholder("float")
        self.W = tf.Variable(tf.random_normal([nX, nY]))
        self.b = tf.Variable(tf.random_normal([nY]))
        self.prediction = tf.add(tf.matmul(self.X, self.W), self.b)
        self.loss       = tf.reduce_sum(tf.abs(tf.reshape(self.prediction,
                                               [-1])- tf.reshape(self.Y, [-1])))/(nY * self.sizeY)
        self.loss_MSE   = tf.reduce_sum(tf.square(tf.reshape(self.prediction,
                                               [-1]) - tf.reshape(self.Y, [-1])))/(nY * self.sizeY)
        
        self.optimizer  = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        self.init       = tf.global_variables_initializer()
     
    # Get model LR with Weights and Biases 
    def launchG(self, trainX, trainY):
        index = -1
        
        with tf.Session() as sess:
            sess.run(self.init)
            for step in range(1, self.num_steps+1):
                index, batch_x, batch_y = cc.get_batch_matrix2d(index, self.batch_size, trainX, trainY)
                # Run optimization op (backprop)
                sess.run(self.optimizer, feed_dict={self.X: batch_x, self.Y: batch_y,
                                                    self.sizeY: self.batch_size, self.keep_prob: self.dropout})
                
           
            W_val, b_val = sess.run([self.W, self.b])
           
        
        # trans to numpy arrays
        W_val = np.float64(W_val)
        b_val = np.float64(b_val)
        
        return (W_val, b_val)
    
      
#defini a class of graph - NN2hiderlayers
class NN2hiddenlayers(object):

    #To build the graph when instantiated
    def __init__(self, nX, nY, n_hidden_1, n_hidden_2):
        
        self.learning_rate = vc.learning_rate
        self.num_steps = vc.nb_epochs
        self.batch_size = vc.batch_size
        self.dropout = vc.keep_prob
        self.keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)
        self.X = tf.placeholder("float", [None, nX])
        self.Y = tf.placeholder("float", [None, nY])
        self.sizeY = tf.placeholder("float")
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2
        
        # Definition of weights & biases
        self.weights = {
            'h1': tf.Variable(tf.random_normal([nX, self.n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2])),
            'out': tf.Variable(tf.random_normal([self.n_hidden_2, nY]))
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
            'out': tf.Variable(tf.random_normal([nY]))
        }
        
        self.prediction = self.neural_net(self.X) 
        self.loss       = tf.reduce_sum(tf.abs(tf.reshape(self.prediction, [-1])
                                 - tf.reshape(self.Y, [-1])))/(nY * self.sizeY)
        self.loss_MSE       = tf.reduce_sum(tf.square(tf.reshape(self.prediction, [-1])
                                 - tf.reshape(self.Y, [-1])))/(nY * self.sizeY)
        self.optimizer  = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        self.init       = tf.global_variables_initializer()
    
    # calcul the prediction
    def neural_net(self, x):
        # Hidden fully connected layer with n_hidden_1 neurons
        layer_1 = tf.add(tf.matmul(x, self.weights['h1']), self.biases['b1'])
        # Hidden fully connected layer with n_hidden_2 neurons
        layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])
        # Output fully connected layer with a neuron for each class
        out_layer = tf.matmul(layer_2, self.weights['out']) + self.biases['out']
        return out_layer
    
    # Get model NN2hiddenlayers with Weights and Biases 
    def launchG(self, trainX, trainY):
        index = -1
        with tf.Session() as sess:
            sess.run(self.init)
            for step in range(1, self.num_steps+1):
                index, batch_x, batch_y = cc.get_batch_matrix2d(index, self.batch_size, trainX, trainY)
                # Run optimization op (backprop)
                sess.run(self.optimizer, feed_dict={self.X: batch_x, self.Y: batch_y,
                                                    self.sizeY: self.batch_size, self.keep_prob: self.dropout})
                
                
            
            weights_val, biases_val = sess.run([self.weights, self.biases])
            
           
        
        return (weights_val, biases_val)


# defini a class of graph - CNN
class CNN(object):

    #To build the graph when instantiated
    def __init__(self, nX, nY, nkernel):
        self.learning_rate = vc.learning_rate
        self.num_steps = vc.nb_epochs
        self.batch_size = vc.batch_size
        self.dropout = vc.keep_prob
        self.nb_kernel = nkernel
        self.keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)
        self.X = tf.placeholder("float", [None, nX])
        self.Y = tf.placeholder("float", [None, nY])
        self.sizeY = tf.placeholder("float")
        # Definition of weights & biases
        self.weights = {
            # 5x5 conv, 1 input, 32 outputs
            'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
            # 5x5 conv, 32 inputs, 64 outputs
            'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
            # fully connected, 64 inputs, 1024 outputs
            'wd1': tf.Variable(tf.random_normal([64, 1024])),
            # 1024 inputs, nY outputs (class prediction)
            'out': tf.Variable(tf.random_normal([1024, nY]))
        }
        self.biases = {
            'bc1': tf.Variable(tf.random_normal([32])),
            'bc2': tf.Variable(tf.random_normal([64])),
            'bd1': tf.Variable(tf.random_normal([1024])),
            'out': tf.Variable(tf.random_normal([nY]))
        }
        
        self.prediction = self.conv_net(self.X, self.weights, self.biases, self.keep_prob)
        self.loss       = tf.reduce_sum(tf.abs(tf.reshape(self.prediction, [-1])
                                               - tf.reshape(self.Y, [-1])))/(nY * self.sizeY)
        self.loss_MSE   = tf.reduce_sum(tf.square(tf.reshape(self.prediction, [-1])
                                               - tf.reshape(self.Y, [-1])))/(nY * self.sizeY)
        self.optimizer  = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        self.init       = tf.global_variables_initializer()
        
    # Create some wrappers for simplicity
    def conv2d(self, x, W, b, strides=1): 
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)


    def maxpool2d(self, x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


    def conv_net(self, x, weights, biases, dropout):
        # data input is a 1-D vector of nX features 
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, daysnumber, Width, Channel]
        nX = (int) (x.shape[1])
        #print(nX)
        x = tf.reshape(x, shape=[-1,  self.nb_kernel, int (nX/self.nb_kernel), 1]) 
        # Convolution Layer
    
        conv1 = self.conv2d(x, weights['wc1'], biases['bc1'])
        # Max Pooling (down-sampling)
        conv1 = self.maxpool2d(conv1, k=2)
        # Convolution Layer
        conv2 = self.conv2d(conv1, weights['wc2'], biases['bc2'])
        # Max Pooling (down-sampling)
        conv2 = self.maxpool2d(conv2, k=2)
        
        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        #fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].shape[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
    
        # Apply Dropout
        fc1 = tf.nn.dropout(fc1, dropout)
        # Output, class prediction
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        return out
    
    # Get model CNN with Weights and Biases
    def launchG(self, trainX, trainY):
        index = -1
        
        with tf.Session() as sess:
            sess.run(self.init)
            for step in range(1, self.num_steps+1):
                index, batch_x, batch_y = cc.get_batch_matrix2d(index, self.batch_size, trainX, trainY)
                # Run optimization op (backprop)
                sess.run(self.optimizer, feed_dict={self.X: batch_x, self.Y: batch_y,
                                                    self.sizeY: self.batch_size, self.keep_prob: self.dropout})
                
            """ 
                if step % 1000 == 0 or step == 1:
                    c = sess.run(self.loss, feed_dict={self.X: batch_x, self.Y: batch_y,
                                                    self.sizeY: self.batch_size, self.keep_prob: self.dropout})
                    print("Step %6d" %step , ", Cost= " + "{:.5f}".format(c))
                
            print("Training Finished!")
            # Calculate loss testing
            c= sess.run(self.loss, feed_dict={self.X: testX, self.Y: testY,
                                              self.sizeY: testY.shape[0],self.keep_prob: self.dropout })
            print("      Testing cost=" , "{:.5f}".format(c))
            
            c_MSE= sess.run(self.loss_MSE, feed_dict={self.X: testX, self.Y: testY,
                                              self.sizeY: testY.shape[0],self.keep_prob: self.dropout })
            print("      Testing cost MSE=" , "{:.5f}".format(c_MSE))
                
            """    
                
            
            weights_val, biases_val = sess.run([self.weights, self.biases])
            
            
            
        return (weights_val, biases_val)
    
     
    
class RNN(object):

    #To build the graph when instantiated
    def __init__(self, nX, nY, timesteps):
        nX = (int)(nX / timesteps)
        self.learning_rate = vc.learning_rate
        self.num_steps = vc.nb_epochs
        self.batch_size = vc.batch_size
        self.dropout = vc.keep_prob
        
        self.timesteps = timesteps
        self.nX = nX
        self.num_hidden = vc.nbhidden
        self.sizeY = tf.placeholder("float")
        self.X = tf.placeholder(tf.float32, [None, self.timesteps, nX])
        self.Y = tf.placeholder(tf.float32, [None, nY])
        self.weights = {
            'out': tf.Variable(tf.random_normal([self.num_hidden, nY]))
        }
        self.biases = {
            'out': tf.Variable(tf.random_normal([nY]))
        }
        self.prediction = self.RNNF(self.X, self.weights,self.biases)
        self.loss       = tf.reduce_sum(tf.abs(tf.reshape(self.prediction, [-1])
                                               - tf.reshape(self.Y, [-1])))/(nY * self.sizeY)
        self.loss_MSE   = tf.reduce_sum(tf.square(tf.reshape(self.prediction, [-1])
                                               - tf.reshape(self.Y, [-1])))/(nY * self.sizeY)
        self.optimizer  = tf.train.AdamOptimizer(learning_rate= self.learning_rate).minimize(self.loss)
        self.init       = tf.global_variables_initializer()

    
    
    def RNNF(self, x, weights, biases):

        # Prepare data shape to match 'rnn' function requirements
        # Current data input shape: (batch_size, timesteps, nX)
        # Required shape: 'timesteps' tensors list of shape (batch_size, nY)
        # Unstack to get a list of 'timesteps' tensors of shape (batch_size, nX)
        x = tf.unstack(x, self.timesteps, 1)

        # Define a lstm cell with tensorflow
        try:
            lstm_cell = rnn.BasicLSTMCell(self.num_hidden, forget_bias=1.0, reuse = False ) # , reuse= True
        except Exception as e:
            lstm_cell = rnn.BasicLSTMCell(self.num_hidden, forget_bias=1.0 , reuse= True) 
        # Get lstm cell output
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], weights['out']) + biases['out']
    
    # Get model RNN with Weights and Biases
    def launchG(self, trainX, trainY):
        index = -1
        
        
        with tf.Session() as sess:
            sess.run(self.init)
            for step in range(1, self.num_steps+1):
                index, batch_x, batch_y = cc.get_batch_matrix2d(index, self.batch_size, trainX, trainY)
                # reshape batch_X for using RNN function
                batch_x = batch_x.reshape((self.batch_size, self.timesteps, self.nX))
                # Run optimization op (backprop)
                sess.run(self.optimizer, feed_dict={self.X: batch_x, self.Y: batch_y,
                                                    self.sizeY: self.batch_size})
                
                
                
            batch_x = batch_x.reshape((self.batch_size, self.timesteps * self.nX))
            
            weights_val, biases_val = sess.run([self.weights, self.biases])
            
        
        
        return (weights_val, biases_val)
        
            
            
class BiRNN(object):

    #To build the graph when instantiated
    def __init__(self, nX, nY, timesteps):
        nX = (int)(nX / timesteps)
        self.learning_rate = vc.learning_rate
        self.num_steps = vc.nb_epochs
        self.batch_size = vc.batch_size
        self.dropout = vc.keep_prob
        self.timesteps = timesteps
        self.nX = nX
        self.num_hidden = vc.nbhidden
        
        self.sizeY = tf.placeholder("float")
        self.X = tf.placeholder(tf.float32, [None, self.timesteps, nX])
        self.Y = tf.placeholder(tf.float32, [None, nY])
        self.weights = {
            'out': tf.Variable(tf.random_normal([2*self.num_hidden, nY]))
        }
        self.biases = {
            'out': tf.Variable(tf.random_normal([nY]))
        }
        
        self.prediction = self.BiRNNF(self.X, self.weights,self.biases)
        self.loss       = tf.reduce_sum(tf.abs(tf.reshape(self.prediction, [-1])
                                               - tf.reshape(self.Y, [-1])))/(nY * self.sizeY)
        self.loss_MSE       = tf.reduce_sum(tf.square(tf.reshape(self.prediction, [-1])
                                               - tf.reshape(self.Y, [-1])))/(nY * self.sizeY)
        self.optimizer  = tf.train.AdamOptimizer(learning_rate= self.learning_rate).minimize(self.loss)
        self.init       = tf.global_variables_initializer()

   
    
    def BiRNNF(self, x, weights, biases):

        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, timesteps, n_input)
        # Required shape: 'timesteps' tensors list of shape (batch_size, num_input)

        # Unstack to get a list of 'timesteps' tensors of shape (batch_size, num_input)
        x = tf.unstack(x, self.timesteps, 1)

        # Define lstm cells with tensorflow
        # Forward direction cell
        lstm_fw_cell = rnn.BasicLSTMCell(self.num_hidden, forget_bias=1.0)
        # Backward direction cell
        lstm_bw_cell = rnn.BasicLSTMCell(self.num_hidden, forget_bias=1.0)

        # Get lstm cell output
        try:
            outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                              dtype=tf.float32)
        except Exception: # Old TensorFlow version only returns outputs not states
            outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                        dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], weights['out']) + biases['out']
    
    # Get model BiRNN with Weights and Biases 
    def launchG(self, trainX, trainY):
        index = -1
        
        
        with tf.Session() as sess:
            sess.run(self.init)
            for step in range(1, self.num_steps+1):
                index, batch_x, batch_y = cc.get_batch_matrix2d(index, self.batch_size, trainX, trainY)
                # reshape batch_X for using RNN function
                batch_x = batch_x.reshape((self.batch_size, self.timesteps, self.nX))
                # Run optimization op (backprop)
                sess.run(self.optimizer, feed_dict={self.X: batch_x, self.Y: batch_y,
                                                    self.sizeY: self.batch_size})
                
                
                
                
            batch_x = batch_x.reshape((self.batch_size, self.timesteps * self.nX))
            
            
            weights_val, biases_val = sess.run([self.weights, self.biases])
            
         
        return (weights_val, biases_val)

           

        

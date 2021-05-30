'''
KerasSV library.

Project: Smart Village
Time series Prediction:
Author:  HO Van Hieu
'''

import numpy as np
import pandas as pd
from smartSV import vocabulariesSV as vc
from smartSV import calculatorSV as cc

from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.wrappers.scikit_learn import KerasClassifier

from keras.layers import LSTM

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline




# definie model CNN
# x_train: data set for training
# out_put_dim: dimentions number  of Y (label set)
def create_CNN_model(x_train, output_dim): 
    
    # create model
    
    num_train, height, width, depth = x_train.shape
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape= (height, width, depth)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))                  
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation='linear'))
    model.add(LeakyReLU(alpha=0.1))  
    model.add(Dropout(0.3))
    model.add(Dense(output_dim, activation = 'softmax'))
    
    model.compile(loss = vc.loss_type, optimizer = vc.optimizer_type )
    
    return model

# definie model RNN
# x_train: data set for training
# out_put_dim: dimentions number  of Y (label set)
def create_RNN_model(x_train, divisor, output_dim):
    
    # create model
    model = Sequential()
    model.add(LSTM(divisor, input_shape=(x_train.shape[1], 1)))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(loss = vc.loss_type, optimizer = vc.optimizer_type)
    
    return model

# Get CNN model 
# trainX: DatasSet for trainning  (Matrix 2D)
# trainY: LabelsSet for trainning
# testX: DatasSet for testing
# Return:
#       + model: a model scikitLearn
#       + predY: Predicting labels set
def CNN_model(trainX, trainY, nb_kernel):
       
    train_X = trainX
    train_Y = trainY
   
    input_dim = train_X.shape[1]
    divisor = nb_kernel
    output_dim = train_Y.shape[1]
    hight = (int)(input_dim/divisor)
    train_X = train_X.reshape((-1, divisor, hight, 1 ))
    
    model = create_CNN_model(train_X, output_dim)
    model.fit(train_X, train_Y, batch_size= vc.batch_size, epochs= vc.nb_epochs, verbose=0)
    
    return model

#Get RNN model
# trainX: DatasSet for trainning  (Matrix 2D)
# trainY: LabelsSet for trainning
# testX: DatasSet for testing
# Return:
#       + model: a model scikitLearn
#       + predY: Predicting labels set
def RNN_model(trainX, trainY, nb_kernel):
    train_X = trainX
    train_Y = trainY
    
    input_dim = train_X.shape[1]
    output_dim = train_Y.shape[1]
    train_X = train_X.reshape((-1, input_dim , 1 ))
    
    model = create_RNN_model(train_X, nb_kernel, output_dim)
    model.fit(train_X, train_Y, batch_size = vc.batch_size, epochs= vc.nb_epochs,verbose=0, shuffle=False)
    
    return model


# CreateLinearReGression model
# input_dim: column number of X set, 
# out_dim: column number of Y set, 
def create_LinearReGression_model(input_dim, output_dim):
    
    # create model, required for KerasClassifier
    model = Sequential()
    model.add(Dense(output_dim = output_dim , input_dim = input_dim , kernel_initializer='normal', activation='relu'))
    # Compile model
    model.compile(loss= vc.loss_type, optimizer= vc.optimizer_type)
    
    return model

# Create NN2hd model
# input_dim: column number of X set, 
# out_dim: column number of Y set, 
def create_NN2hd_model(input_dim, output_dim):
    
    # create model, required for KerasClassifier
    model = Sequential()
    model.add(Dense(input_dim + 5, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
    model.add(Dense(output_dim + 5, kernel_initializer='normal', activation='relu'))
    model.add(Dense(output_dim, kernel_initializer='normal'))
    # Compile model
    model.compile(loss = vc.loss_type, optimizer = vc.optimizer_type)
    
    return model
    
#Get Regression model: predY: predict of testX set
# method is in methods list "methods"
# train_X: DatasSet for trainning  (Matrix 2D)
# train_Y: LabelsSet for trainning 
# test_X: DatasSet for testing 
# Return:
#       + model: a model scikitLearn
#       + pred_Y: Predicting labels set 
def Regression_model(trainX, trainY, method):
    
    input_dim = trainX.shape[1]
    output_dim = trainY.shape[1]
    
    if method == vc.NN2hd:
        model = create_NN2hd_model(input_dim, output_dim)
    else:
        model = create_LinearReGression_model(input_dim, output_dim)

    model.fit(trainX, trainY, batch_size = vc.batch_size, epochs= vc.nb_epochs,verbose=0)
    
    return model


#defini a class model
class model_keras(object):
    #To build the model when instantiated
    # method: name of method (CNN, RNN ..?)
    # trainX: DataSet for training
    # trainY: LabelSet for training
    def __init__(self, method, trainX, trainY):
        self.method = method
        self.trainX = trainX
        self.trainY = trainY
        self.nX = trainX.shape[1]
        self.nY = trainY.shape[1]
        self.nb_kernel =  cc.choose_divisor(self.nX)
        self.model_method = self.set_model_method(self.trainX, self.trainY)
    
    # to buil model (training phase)
    # trainX: DataSet for training - numpyarray
    # trainY: LabelSet for training - numpyarray
    def set_model_method(self, trainX, trainY):
        if( self.method in [vc.LinearRegression, vc.NN2hd] ):
             model_method = Regression_model(trainX, trainY,  self.method)
        elif (self.method == vc.CNN):
             model_method= CNN_model(trainX, trainY, self.nb_kernel)
        elif (self.method == vc.RNN):
             model_method= RNN_model(trainX, trainY, self.nb_kernel)
        
        else:
            model_method = None
            
        return model_method
    
    # get the prediction
    # Xtest: DataSet - numpyarray
    # return: the prediction - numpyarray
    def get_predict(self, Xtest):
        if( self.method == vc.LinearRegression):
            predict = self.model_method.predict(Xtest, verbose=0)
        
        elif (self.method == vc.NN2hd):
            predict = self.model_method.predict(Xtest, verbose=0)
        elif (self.method == vc.CNN):
            hight = (int)(self.nX/self.nb_kernel)
            testX = Xtest.reshape((-1, self.nb_kernel, hight, 1 ))
            predict = self.model_method.predict(testX, verbose=0)
        elif (self.method == vc.RNN):
            testX = Xtest.reshape((-1, self.nX , 1 ))
            predict = self.model_method.predict(testX, verbose=0)
        
        else:
            predict = None
        
        return predict
    

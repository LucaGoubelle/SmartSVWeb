'''
PyTorchSV library.
Project: Smart Village
Prediction
Author:  HO Van Hieu
'''

import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
import calculatorSV as cc
import calculatoPytorchSV as ccpt
import vocabulariesSV as vc
import math, random
import torch.nn.functional as F

torch.set_default_tensor_type('torch.DoubleTensor')


# Define Linear Regression kernel 
class LinearRegression(nn.Module):
    
    #nX: number of columns of DataSet
    #nY: number of columns of LabelSet
    def __init__(self,  nX, nY):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(nX, nY)  # input and output is nX * nY dimension

    
    def forward(self, trainX):
        out = self.linear(trainX)
        return out

# Get a Linear Regression model 
# trainX: Data Set for training
# trainY: Label Set for training
# Return: a Linear Regression model 
def LinearRegression_model(trainX, trainY):

    nX = trainX.shape[1]
    nY = trainY.shape[1]
    model = LinearRegression(nX, nY)
    # define Loss
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=vc.learning_rate)
       
    index = -1
        
    x_train = torch.from_numpy(trainX)
    y_train = torch.from_numpy(trainY)
    
    for epoch in range(vc.nb_epochs):
        
        # forward
        out = model(Variable(x_train))
        loss = criterion(out, Variable(y_train))
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    
        
    return model



# Fully connected neural network with 2 hiddens layer
# input_size: number of columns of DataSet
# hidden1_size: number of columns of hidden1
# hidden2_size: number of columns of hidden2
# output_size: number of columns of LabelSet
class NN2hd(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super(NN2hd, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size) 
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2_size, output_size)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

# Get a NN2hd model 
# trainX: Data Set for training
# trainY: Label Set for training
# Return: a NN2hd model 
def NN2hd_model(x_train, y_train):
    input_size = x_train.shape[1]
    output_size = y_train.shape[1]
    hidden1_size = input_size + 5
    hidden2_size = output_size +5
    
    model = NN2hd(input_size, hidden1_size, hidden2_size, output_size)

    # Loss and optimizer
    criterion = nn.MSELoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=vc.learning_rate)  
    optimizer = optim.SGD(model.parameters(), lr=vc.learning_rate)
    index = -1
        
    # Train the model
    for epoch in range(vc.nb_epochs):
        index, batch_x, batch_y = cc.get_batch_matrix2d(index, vc.batch_size, x_train, y_train)
        
        
        b_x = np.array(batch_x)
        b_y = np.array(batch_y)
        b_x =  torch.DoubleTensor(b_x)   
        b_y =  torch.DoubleTensor(b_y)
        
        # Forward pass
                
        outputs = model(Variable(b_x))
        loss = criterion(outputs, Variable(b_y))
              
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

    model.eval()
    
    return model

# Define CNN kernel  
class CNN(nn.Module):
    
    
    # nx * timesteps: number of columns of DataSet
    # output_size: number of columns of labelSet
    # def __init__(self, batch_size, nx, timesteps,  output_size): # # batch_size: number of records of DataSet
    def __init__(self, nx, timesteps,  output_size): # 
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, input_size1, input_size2)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this  shape after con2d, padding=(kernel_size-1)/2 if stride=1
                ),                              # output shape (16, xx, xx)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, xx, xx)
        )
        self.conv2 = nn.Sequential(         # input shape (16, )
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, )
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),                # output shape (32, )
        )
        
        self.nx =nx
        self.timesteps = timesteps
        sizeIP = cc.choose_divisor(timesteps)
        self.out = nn.Linear(32 *  (int)(timesteps/sizeIP) *  1, output_size)   # fully connected layer, output 10 classes
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * *)
        output = self.out(x)
        return output
    

# Get a CNN model 
# x_train: Data Set for training
# y_train: Label Set for training
# Return: a CNN model     
def CNN_model(x_train, y_train):
    input_size = x_train.shape[1]
    nx = cc.choose_divisor(input_size)
    timesteps = (int)(input_size / nx)
    output_size = y_train.shape[1]
        
    model = CNN(nx, timesteps, output_size)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=vc.learning_rate)  
    
    index = -1
    
    # Train the model
    for epoch in range(vc.nb_epochs):
        index, batch_x, batch_y = cc.get_batch_matrix2d(index, vc.batch_size, x_train, y_train)
        batch_x = batch_x.reshape((vc.batch_size, 1, timesteps, nx))
        b_x = np.array(batch_x)
        b_y = np.array(batch_y)
        b_x =  torch.DoubleTensor(b_x)   
        b_y =  torch.DoubleTensor(b_y)
        
        # Forward pass
                
        outputs = model(Variable(b_x))
        loss = criterion(outputs, Variable(b_y))
              
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    model.eval()
    
    return model


# Define RNN kernel  
class RNN(nn.Module):
    
    # input_size: number of columns of DataSet
    # num_classes: number of columns of labelSet
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial states 
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) 
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        # Forward propagate RNN
        out, _ = self.lstm(x, (h0, c0))  

        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])  
        return out
    

# Get a RNN model 
# x_train: Data Set for training
# y_train: Label Set for training
# Return: a RNN model 
def RNN_model(x_train, y_train):
    input_size = cc.choose_divisor(x_train.shape[1])
    hidden_size = 16
    output_size = y_train.shape[1]
    
    timesteps = (int) (x_train.shape[1] / input_size)
        
    model = RNN(input_size, hidden_size, timesteps,  output_size)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=vc.learning_rate)  
    
    index = -1
        
    # Train the model
    for epoch in range(vc.nb_epochs):
        index, batch_x, batch_y = cc.get_batch_matrix2d(index, vc.batch_size, x_train, y_train)
        b_x = np.array(batch_x)
        b_y = np.array(batch_y)
        
        b_x = b_x.reshape(-1, timesteps, input_size)
        b_x =  torch.DoubleTensor(b_x)   
        b_y =  torch.DoubleTensor(b_y)
        
        # Forward pass
                
        outputs = model(Variable(b_x))
        loss = criterion(outputs, Variable(b_y))
                      
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    
    model.eval()
    
    return model


#defini a class model
class model_pytorch(object):
    #To build the model when instantiated
    # method: name of method (CNN, RNN ..?)
    # trainX: DataSet for training
    # trainY: LabelSet for training
    def __init__(self, method, trainX, trainY):
        self.method = method
        self.model_method = self.set_model_method(trainX, trainY)
    
    # to buil model (training phase)
    # trainX: DataSet for training - numpyarray
    # trainY: LabelSet for training - numpyarray
    def set_model_method(self, trainX, trainY):
        if( self.method == vc.LinearRegression):
            model= LinearRegression_model(trainX, trainY)
        elif (self.method == vc.NN2hd):
            model= NN2hd_model(trainX, trainY)
        elif (self.method == vc.CNN):
            model= CNN_model(trainX, trainY)
        elif (self.method == vc.RNN):
            model= RNN_model(trainX, trainY)
        else:
            model = None
    
        return model
    
    # get the prediction
    # Xtest: DataSet - numpyarray
    # return: the prediction - numpyarray
    def get_predict(self, Xtest):
        if( self.method == vc.LinearRegression):
            predict = ccpt.getPredict_LinearRegression_Pytorch(self.model_method , Xtest)
        elif (self.method == vc.NN2hd):
            predict = ccpt.getPredict_NN2hiddenlayers_Pytorch(self.model_method , Xtest)
        elif (self.method == vc.CNN):
            predict = ccpt.getPredict_CNN_Pytorch(self.model_method , Xtest)
        elif (self.method == vc.RNN):
            predict = ccpt.getPredict_RNN_Pytorch(self.model_method , Xtest)
        else:
            predict = None
    
        return predict 
    



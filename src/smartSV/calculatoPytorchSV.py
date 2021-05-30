"""
project: Smart Village
calculatoPytorchSV
Author: HO Van Hieu
"""

import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable

# Predict from LinearRegression model
# model: a PytorchSV - LinearRegression model 
# Xtest: Data set for prediction
# Return: the prediction - numpy array 
def getPredict_LinearRegression_Pytorch(model, Xtest):
    Xtest = torch.from_numpy(Xtest)
    Xtest = torch.DoubleTensor(Xtest)
    predict = model(Variable(Xtest))
    predict = predict.data.numpy()
    predict = np.array(predict, dtype=np.float32)
    return predict

# Predict from NN2hiddenlayers model
# model: a PytorchSV - NN2hiddenlayers model 
# Xtest: Data set for prediction
# Return: the prediction - numpy array 
def getPredict_NN2hiddenlayers_Pytorch(model, Xtest):
    Xtest = torch.from_numpy(Xtest)
    Xtest = torch.DoubleTensor(Xtest)
    predict = model(Variable(Xtest))
    predict = predict.data.numpy()
    predict = np.array(predict, dtype=np.float32)
    return predict

# Predict from CNN model
# model: a PytorchSV - CNN model 
# Xtest: Data set for prediction
# Return: the prediction - numpy array 
def getPredict_CNN_Pytorch(model, Xtest):
    
    xtest = Xtest.reshape((-1, 1, model.timesteps, model.nx))
    xtest = torch.from_numpy(xtest)
    xtest = torch.DoubleTensor(xtest)
    predict = model(Variable(xtest))
    predict = predict.data.numpy()
    predict = np.array(predict, dtype=np.float32)
    return predict

# Predict from RNN model
# model: a PytorchSV - RNN model 
# Xtest: Data set for prediction
# Return: the prediction - numpy array 
def getPredict_RNN_Pytorch(model, Xtest):
    
    xtest = Xtest.reshape((-1, model.num_layers, model.input_size))
    xtest = torch.from_numpy(xtest)
    xtest = torch.DoubleTensor(xtest)
    predict = model(Variable(xtest))
    predict = predict.data.numpy()
    predict = np.array(predict, dtype=np.float32)
    
    return predict



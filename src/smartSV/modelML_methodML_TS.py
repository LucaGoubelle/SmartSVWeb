
'''
project: Smart Village
Classes and fonctions for forecasting Time Series
Author: HO Van Hieu
'''

# import necessary librairies 
import sys
import numpy as np
import time
import copy
import datetime as dt
from numpy  import array

import processDataSV as pd
import calculatorSV as cc
import vocabulariesSV as vc
from predictModelSV import PredictModel as pm




class MethodML_TS(object):
    """
    A Machine Learning Method for Time Series
    
    Define a Machine Learning method for the forecasting Time series
    - nb_previousDays: number of previous days. use information of nb_previousDays
                     to forecast the nb_nextDays
    - libraryName: Name of Machine LEarning library (scikit-learn, keras, ...)
    - methodName: Name of Machine Learning method (LinearRegresssion, ...)
    """
  
    def __init__(self, nb_previousDays, nb_nextDays, libraryName, methodName):
        self.nb_previousDays = nb_previousDays
        self.nb_nextDays = nb_nextDays
        self.libraryName = libraryName
        self.methodName = methodName
    
    def get_nb_nextDays(self):
        return self.nb_nextDays
    def get_nb_previousDays(self):
        return self.nb_previousDays
    def get_libraryName(self):
        return self.libraryName
    def get_methodName(self):
        return self.methodName
    
    def get_infos(self):
        return (self.nb_previousDays,
                self.nb_nextDays,                
                self.libraryName,
                self.methodName)
    def print_(self):
        print(  self.nb_previousDays,
                self.nb_nextDays,
                self.libraryName,
                self.methodName)
    
    def set_nb_nextDays(self, nb_nextDays):
        self.nb_nextDays = nb_nextDays
    def set_nb_previousDays(self, nb_previousDays):
        self.nb_previousDays = nb_previousDays
    def set_libraryName(self, libraryName):
        self.libraryName = libraryName
    def set_methodName(self, methodName):
        self.methodName = methodName


# Define a Machine Learning Model for the forecasting TS
      
class ModelML_TS(object):
    """
    A Machine Learning Model for forecasting Time Series
    - methodML_DER: Machine Learning method
    - model: Machine LEarning model
    """
  
    def __init__(self, methodML_TS, model):
        self.methodML_TS = methodML_TS
        self.model = model


    def get_methodML(self):
        return self.methodML_TS
    def get_model(self):
        return self.model
    
    def set_methodML(self, methodML_TS):
        self.methodML_TS = methodML_TS
    def set_model(self, model):
        self.model = model


       
        


def get_methodML_TSs(list_previousDays, list_nextDays,  methods):
    """
    Product a list of Machine Learning methods for forecasting Time Series
    - list_methods: list of methods (method[0]: library name,
                                     method[1]: machine method name)
    - Return:
        +list of Machine Learning methods for forecasting Time Series
    """
    
    methodsTS = []
    for previousDays in list_previousDays:
        for nextDays in list_nextDays:
            for method in methods:
                methodML = MethodML_TS(previousDays, nextDays, method[0], method[1])
                methodsTS.append(copy.deepcopy(methodML))
    
    return methodsTS



def ExcutationML_TS(trainX, trainY, testX, testY, methodML_TS):
    """
    Do Training and Testing steps for get the model, loss, and calculate time
    - trainX, trainY: Training Sets (X-> Y)
    - testX, testY: Training Sets (X-> Y)
    - methodML_TL: machine learning method
    - Return:
        + modelML_EDR: Machine LEarning model
        + loss: between predict set and testY set
        + calcul_time: calculate time
    """
    
    model = pm(methodML_TS.get_libraryName(), methodML_TS.get_methodName())
    time_begin = time.time()
    model = model.set_model(trainX, trainY)
    predict = model.get_predict(testX)
    time_end = time.time()
    calcul_time = time_end - time_begin
    loss = cc.get_loss_MSE(predict, testY)
    modelML_TS = ModelML_TS(methodML_TS, model)
    
    return modelML_TS, loss, calcul_time        
        
        

def get_model_and_loss_TS_from_matrix(matrix, columnsX,
                                      columnsY, percentTrain,
                                      methodML_TS):
    
    """
    Produce Machine Learning model, and corresponding loss value
    - matrix: data matrix
    - columnsX: list of columns of X
    - columnsY: list of columns of Y
    - perscentTrain: percent of Training set (training set / data)
    - methodML_TS: machine learning method
    - Return:
        + modelML_TS: Machine LEarning model
        + loss: the difference between predict_set and testY_set
        + calcul_time: calculate time
    """
    nb_previousDays = methodML_TS.get_nb_previousDays()
    nb_nextDays = methodML_TS.get_nb_nextDays()
    X, Y = pd.productXY_timeseries_from_matrix(matrix, columnsX,
                              columnsY, nb_previousDays, nb_nextDays)

    # produce trainning set and testing set
    trainX, trainY, testX, testY = cc.divise_TrainingSet_TestingSet(X, Y, percentTrain)
    modelML_TS, loss, calcul_time = ExcutationML_TS(trainX, trainY,
                                                    testX, testY,
                                                    methodML_TS)
    
    return (modelML_TS, loss, calcul_time)


def get_list_model_and_loss_TS_from_matrix(matrix, columnsX, columnsY,
                                           methodML_TSs, percentTrain):
    
    """
    Produce the list of (Machine Learning model, and corresponding loss value)
    - matrix: data matrix
    - columnsX: list of columns of X
    - columnsY: list of columns of Y
    - methodML_TSs: list of Machine Learning methods for forecasting Time series , 
    - perscentTrain: percent of Training set (training set / data)
    - Return:
        + L: List of (Machine Learning model, and corresponding loss value)
             (loss = difference between predict_set and testY_set)
        + timeL: execution time list
    
    """
    
    L = []
    timeL = []
    for methodML_TS in methodML_TSs:
        modelML_TS, loss, calcul_time  = get_model_and_loss_TS_from_matrix(matrix,
                                                                            columnsX,
                                                                            columnsY,
                                                                            percentTrain,
                                                                            methodML_TS)
        L.append([copy.deepcopy(modelML_TS), copy.deepcopy(loss)])
        timeL.append(calcul_time)
    return L, timeL




def get_best_model_TS_from_matrix(matrix, columnsX, columnsY, methodML_TSs, percentTrain):
    """
     Model of ML which has the lowest loss value (best model)
    - matrix: data matrix
    - columnsX: list of columns of X
    - columnsY: list of columns of Y
    - methodML_TSs: list of Machine Learning methods for forecasting Time series , 
    - perscentTrain: percent of Training set (training set / data)
    - Return:
        + model of ML which has the lowest loss value (best model)
    """
    
    
    L, timeL = get_list_model_and_loss_TS_from_matrix(matrix, columnsX, columnsY,
                                                      methodML_TSs, percentTrain)
    # search index of item minimal in the L list.
    indexMin = pd.searchIndex_Min_inColumn_inMatrix2D(L, 1)
    return L[indexMin][0], L[indexMin][1]
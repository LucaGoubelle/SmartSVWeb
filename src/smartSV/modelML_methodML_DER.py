'''
project: Smart Village
Classes and fonctions for predictionDER model 
Author: HO Van Hieu
'''

# import necessary librairies 
import sys
import numpy as np
import time
import copy
import datetime as dt
from numpy  import array

from smartSV import processDataSV as pd
from smartSV import calculatorSV as cc
from smartSV import vocabulariesSV as vc
from smartSV.predictModelSV import PredictModel as pm




class MethodML_DER(object):
    """
    A Machine Learning Method
    Define a Machine Learning method for the forecasting DE-> R
    - libraryName: Name of Machine LEarning library (scikit-learn, keras, ...)
    - methodName: Name of Machine Learning method (LinearRegresssion, ...)
    """
  
    def __init__(self, libraryName, methodName):
        self.libraryName = libraryName
        self.methodName = methodName

    def get_libraryName(self):
        return self.libraryName
    def get_methodName(self):
        return self.methodName
    
        
    def set_libraryName(self, libraryName):
        self.libraryName = libraryName
    def set_methodName(self, methodName):
        self.methodName = methodName
    
    def print_(self):
        print(self.libraryName, ";", self.methodName)
    


class ModelML_DER(object):
    """
    Define a Machine Learning Model for the forecasting DER
    - methodML_DER: Machine Learning method
    - model: Machine LEarning model
    """
    
    def __init__(self, methodML_DER, model):
        self.methodML_DER = methodML_DER
        self.model = model


    def get_methodML_DER(self):
        return self.methodML_DER
    def get_model(self):
        return self.model
    
    def set_methodML_DER(self, methodML_DER):
        self.methodML_DER = methodML_DER
    def set_model(self, model):
        self.model = model
        


def get_methodsList_DER(list_methods):
    """
    Producce a list of Machine Learning methods for forecasting DE-> R
    - list_methods: list of methods (method[0]: library name,
                                     method[1]: machine method name)
    Return:
        +list of Machine Learning methods for forecasting DE-> R
    """
    methodsMLList = []
    for method in list_methods:
        methodML = MethodML_DER(method[0], method[1])
        methodsMLList.append(methodML)
    
    return methodsMLList



def ExcutationML_DER(trainX, trainY, testX, testY, methodML_DER):
    """
    Do Training and Testing steps for get the model, loss, and calculate time
    - trainX, trainY: Training Sets (X-> Y)
    - testX, testY: Training Sets (X-> Y)
    - methodML_DER: machine learning method
    Return:
        + modelML_EDR: Machine LEarning model
        + loss: between predict set and testY set
        + calcul_time: calculate time
    """
    model = pm(methodML_DER.get_libraryName(), methodML_DER.get_methodName())
    time_begin = time.time()
    model = model.set_model(trainX, trainY)
    predict = model.get_predict(testX)
    time_end = time.time()
    calcul_time = time_end - time_begin
    loss = cc.get_loss_MSE(predict, testY)
    modelML_DER = ModelML_DER(methodML_DER, model)
    
    return modelML_DER, loss, calcul_time



def get_model_and_loss_DER(path_file, listColumn_DE, listColumn_R, percentTrain, methodML_DER):
    """
    Produce the list of (Machine Learning model, and corresponding loss value)
    - path_file: path of data file
    - listColumn_DE: column indexes  of DE data in csv data file 
    - listColumn_R: column indexes of R data in csv data file 
    - perscentTrain: percent of Training set (training set / data)
    - methodML_DER: machine learning method
    Return:
        + modelML__DER: Machine LEarning model (DE -> R)
        + loss: the difference between predict_set and testY_set
        + calcul_time: calculate time
    """
    dataframe = pd.read_csv(path_file, delimiter = vc.delimit, begining_line_number = 1)
    X, Y = pd.productXY_DER_from_matrix(dataframe.values, listColumn_DE, listColumn_R)

    # product trainning set and testing set
    trainX, trainY, testX, testY = cc.divise_TrainingSet_TestingSet(X, Y, percentTrain)
    modelML_DER, loss, calcul_time = ExcutationML_DER(trainX, trainY, testX, testY, methodML_DER)
    
    return (modelML_DER, loss, calcul_time)





def get_list_model_and_loss_DER(path_file,listColumn_DE, listColumn_R, methodsList_DER, percentTrain):
    """
    Produce the list of (Machine Learning model, and corresponding loss value)
    - path_file: path of data file
    - listColumn_DE: column indexes  of DE data in csv data file 
    - listColumn_R: column indexes of R data in csv data file 
    - methodsList_DER: list of methods _DER, 
    - perscentTrain: percent of Training set (training set / data)
    Return:
        + L: List of (Machine LEarning model Ei, corresponding loss value)
        (difference between predict_set and testY_set) 
        + timeL: list of calculate time
    """
    L = []
    timeL = []
    for item in methodsList_DER:
        modelML_DER, loss, calcul_time  = get_model_and_loss_DER(path_file, listColumn_DE, listColumn_R, percentTrain, item)
        L.append([copy.deepcopy(modelML_DER), copy.deepcopy(loss)])
        timeL.append(calcul_time)
    return L, timeL


def get_best_model_DER(path_file, listColumn_DE, listColumn_R, methodsList_DER, percentTrain):
    """
    Produce the ML Model which has the lowest loss value (best model)
    - path_file: path of data file
    - listColumn_DE: column indexes  of DE data in csv data file 
    - listColumn_R: column indexes of R data in csv data file 
    - methodsList_DER: list of methods DER,  
    - perscentTrain: percent of Training set (training set / data)
    Return:
        + model of ML which has the lowest loss value (best model)
    """
    L, timeL = get_list_model_and_loss_DER(path_file, listColumn_DE, listColumn_R, methodsList_DER, percentTrain)
    # search index of item minimal in the L list.
    indexMin = pd.searchIndex_Min_inColumn_inMatrix2D(L, 1)
    return L[indexMin][0]




def get_model_and_loss_DER_from_XY(X, Y, percentTrain, methodML_DER):
    
    """
    Produce Machine Learning model, and corresponding loss value
    - perscentTrain: percent of Training set (training set / data)
    - methodML_EDR: machine learning method
    Return:
        + modelML_DER: Machine LEarning model (DE -> R)
        + loss: the difference between predict_set and testY_set
        + calcul_time: calculate time
    """
    
    # produce trainning set and testing set
    trainX, trainY, testX, testY = cc.divise_TrainingSet_TestingSet(X, Y, percentTrain)
    modelML_DER, loss, calcul_time = ExcutationML_DER(trainX, trainY, testX, testY, methodML_DER)
    
    return (modelML_DER, loss, calcul_time)



def get_list_model_and_loss_DER_from_XY(X, Y, methodsList_DER, percentTrain):
    """
    Produce the list of (Machine Learning model, and corresponding loss value)
    - methodsList_DER: list of methods _DER, 
    - perscentTrain: percent of Training set (training set / data)
    Return:
        + L: List of (Machine Learning model, and corresponding loss value)
             (loss = difference between predict_set and testY_set)
        + timeL: list of calculate time
    """
    
    L = []
    timeL = []
    for item in methodsList_DER:
        modelML_DER, loss, calcul_time  = get_model_and_loss_DER_from_XY(X, Y, percentTrain, item)
        L.append([copy.deepcopy(modelML_DER), copy.deepcopy(loss)])
        timeL.append(calcul_time)
    return L, timeL



def get_best_model_DER_from_XY(X, Y, methodsList_DER, percentTrain):
    """
    Find Best Model of ML which has the lowest loss value
    - methodsList_DER: list of methods _DER,  
    - perscentTrain: percent of Training set (training set / data)
    Return:
        + model of ML which has the lowest loss value (best model)
    """
    L, timeL = get_list_model_and_loss_DER_from_XY(X,Y, methodsList_DER, percentTrain)
    # search index of item minimal in the L list.
    indexMin = pd.searchIndex_Min_inColumn_inMatrix2D(L, 1)
    
    return L[indexMin][0]

'''
project: Smart Village
Classes and fonctions for predictionEi model 
Author: HO Van Hieu
'''

# import necessary librairies 
import sys
import time
import copy
import numpy as np
import datetime as dt
from numpy  import array

import processDataSV as pd
import calculatorSV as cc
import vocabulariesSV as vc
from predictModelSV import PredictModel as pm



class MethodML_Ei(object):
    """
    A Machine Learning Method
    Define a Machine Learning method for the forecasting Environmental Information
    - nb_previousDays: number of previous days. 
                       use information of nb_previousDays to forecast the next day
    - libraryName: Name of Machine LEarning library (scikit-learn, keras, ...)
    - methodName: Name of Machine Learning method (LinearRegresssion, ...)
    """
  
    def __init__(self, nb_previousDays,libraryName, methodName):
        self.nb_previousDays = nb_previousDays
        self.libraryName = libraryName
        self.methodName = methodName

    def get_nb_previousDays(self):
        return self.nb_previousDays
    def get_libraryName(self):
        return self.libraryName
    def get_methodName(self):
        return self.methodName
    
    def set_nb_previousDays(self, nb_previousDays):
        self.nb_previousDays = nb_previousDays
    def set_libraryName(self, libraryName):
        self.libraryName = libraryName
    def set_methodName(self, methodName):
        self.methodName = methodName



class ModelML_Ei(object):
    """
    A Machine Learning Model
    Define a Machine Learning Model for the forecasting Environmental Information
    - methodML_Ei: Machine Learning method
    - model: Machine LEarning model
    """
  
    def __init__(self, methodML_Ei, model):
        self.methodML_Ei = methodML_Ei
        self.model = model


    def get_methodML(self):
        return self.methodML_Ei
    def get_model(self):
        return self.model
    
    def set_methodML(self, methodML_Ei):
        self.methodML_Ei = methodML_Ei
    def set_model(self, model):
        self.model = model
        


def get_methodsList_Ei(list_previousDays, list_methods):
    """
    Produce a list of Machine Learning methods for forecasting of Environmental database
    - list_previousDays: list of previous days number
    - list_methods: list of methods (method[0]: library name, method[1]: machine method name)
    - Return:
        + list of methods MethodML_Ei
    """
    methodsEiList = []
    for previousDays in list_previousDays:
        for method in list_methods:
            methodML = MethodML_Ei(previousDays, method[0], method[1])
            methodsEiList.append(copy.deepcopy(methodML))
    
    return methodsEiList



def ExcutationML_Ei(trainX, trainY, testX, testY, methodML_Ei):
    """
    Do Training and Testing steps for get the model, loss, and calculate time
    - trainX, trainY: Training Sets (X-> Y)
    - testX, testY: Training Sets (X-> Y)
    - methodML_Ei: machine learning method
    - Return:
        + modelML_Ei: Machine LEarning model
        + loss: the difference between predict_set and testY_set
        + calcul_time: calculate time
    """
    model = pm(methodML_Ei.get_libraryName(), methodML_Ei.get_methodName())
    time_begin = time.time()
    model = model.set_model(trainX, trainY)
    predict = model.get_predict(testX)
    time_end = time.time()
    calcul_time = time_end - time_begin
    loss = cc.get_loss_MSE(predict, testY)
    modelML_Ei = ModelML_Ei(methodML_Ei, model)
    
    return modelML_Ei, loss, calcul_time



def get_model_and_loss_Ei(path_file, idColumn, percentTrain, methodML_Ei):
    """
    Produce Machine Learning model, and corresponding loss value
    - path_file: path of data file
    - idColumn: column index in csv file "path_file"
    - perscentTrain: percent of Training set (training set / data)
    - Return:
        + modelML_Ei: Machine LEarning model
        + loss: the difference between predict_set and testY_set
        + calcul_time: calculate time
    """
    arrColumn_examples=[idColumn]
    arrColumn_lables = [idColumn]
    number_date_data = methodML_Ei.get_nb_previousDays()
    number_date_predict = 1
    
    
    dataframe = pd.read_csv(path_file, delimiter = vc.delimit, begining_line_number = 1)
    X, Y = pd.productXY_timeseries_from_matrix(dataframe.values,
                                               arrColumn_examples, 
                                               arrColumn_lables,
                                               number_date_data,
                                               number_date_predict)

    # produce trainning set and testing set
    trainX, trainY, testX, testY = cc.divise_TrainingSet_TestingSet(X, Y, percentTrain)
    modelML_Ei, loss, calcul_time = ExcutationML_Ei(trainX, trainY, testX, testY, methodML_Ei)
    
    return (modelML_Ei, loss, calcul_time)


def get_list_model_and_loss_Ei(path_file, idColumn, methodsList_Ei, percentTrain):
    """
    Produce the list of (Machine Learning model, and corresponding loss value)
    - path_file: path of data file
    - idColumn: column index in csv file "path_file"
    - methodsList_Ei: list of methods Ei, 
    - perscentTrain: percent of Training set (training set / data)
    - Return:
        + L: List of (Machine Learning model, and corresponding loss value)
             (loss = difference between predict_set and testY_set)
        + timeL: list of calculate time
    """
    
    L = []
    timeL = []
    for item in methodsList_Ei:
        modelML_Ei, loss,calcul_time  = get_model_and_loss_Ei(path_file, idColumn, percentTrain, item)
        L.append([copy.deepcopy(modelML_Ei), copy.deepcopy(loss)])
        timeL.append(calcul_time)
    return L, timeL




def get_best_model_Ei(path_file, idColumn, methodsList_Ei, percentTrain):
    """
    Find Best Model of ML which has the lowest loss value
    - path_file: path of data file
    - idColumn: column index in csv file "path_file"
    - methodsList_Ei: list of methods Ei, 
    - perscentTrain: percent of Training set (training set / data)
    - Return:
        + model of ML which has the lowest loss value(best model)
    """
    L, timeL = get_list_model_and_loss_Ei(path_file, idColumn, methodsList_Ei, percentTrain)
    # search index of item minimal in the L list.
    indexMin = pd.searchIndex_Min_inColumn_inMatrix2D(L, 1)
    return L[indexMin][0]



def produce_prediction_E(listModelEi, XlastTr):
    """
    Producce the prediction for Environmental datas
    - listModelEi: list of Machine Learning model
    - XlastTr: a matrix 2D. we use XlastTr[i] for forecasting the Ei_nextValue
    - Return: 
        + prediction of Environmental dataSet
    """
    predict_E = []
    index = 0
    for model_iE in listModelEi:
        nbpredays = (model_iE.get_methodML()).get_nb_previousDays()
        XlastTriE = np.array(XlastTr[index])
        XlastTriE = np.reshape(XlastTriE, (-1))
        #print("XlastTriE", XlastTriE)
        lenX = len(XlastTriE)
        #print("lenT", lenX)
        XlastTriE = XlastTriE[lenX - nbpredays:]
        #print("XlastTriE", XlastTriE)
        #print(np.matrix(XlastTriE))
        predict_iE = model_iE.get_model().get_predict(np.matrix(XlastTriE))
        predict_E.append(predict_iE[0])
        index += 1
    predict_E = np.squeeze(np.asarray(predict_E))   
    return predict_E



   





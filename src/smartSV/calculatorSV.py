"""
project: Smart Village
CalculatorSV librairy
Author: HO Van Hieu
"""

import numpy as np
from smartSV import vocabulariesSV as vc
import math


# Select the divisor number in the list divisors
# Ex: input 14, output 7
def choose_divisor(numberInt):
    for v in vc.divisors:
        if np.logical_and((np.mod(numberInt,  v)==0),(np.divide(numberInt,  v) > 1)):
            return v
    return 1

# product trainingSet and of testingSet
# Xset: dataSet
# Yset: labelSet
# percent: percentage of TrainingSet number
# Return: 4 set (Train: Xtrain, Ytrain; Test: Xtest, Ytest)

def divise_TrainingSet_TestingSet(Xset, Yset, percent):
    
    nbXset = Xset.shape[0]
    nbTrainingSet = (int) (nbXset * percent / 100)
    nbMaxTestingSet = nbXset - (nbTrainingSet - 1)
    nbTestingSet = nbMaxTestingSet
    
    XtrainSet = Xset[0: nbTrainingSet]
    YtrainSet = Yset[0: nbTrainingSet]
    XtestSet = Xset[nbTrainingSet: nbTrainingSet+ nbTestingSet]
    YtestSet = Yset[nbTrainingSet: nbTrainingSet+ nbTestingSet]
    
    return (XtrainSet, YtrainSet, XtestSet, YtestSet)

# assure the number of Y set  = 2^n
def divise_TrainingSet_TestingSet2(Xset, Yset, percent):
    
    nbXset = Xset.shape[0]
    nbTrainingSet = (int) (nbXset * percent / 100)
    nbMaxTestingSet = nbXset - (nbTrainingSet - 1)
    nbTestingSet = 2
    while ((nbTestingSet * 2) <=  nbMaxTestingSet):
        nbTestingSet *= 2
    
    XtrainSet = Xset[0: nbTrainingSet]
    YtrainSet = Yset[0: nbTrainingSet]
    XtestSet = Xset[nbTrainingSet: nbTrainingSet+ nbTestingSet]
    YtestSet = Yset[nbTrainingSet: nbTrainingSet+ nbTestingSet]
    
    return (XtrainSet, YtrainSet, XtestSet, YtestSet)

# get loss absolute for the vectors (Y is a list of number)
def get_loss_vector(Ypre, Y):
        resul = np.sum(np.absolute(Ypre - Y))
        return resul/len(Y)

# get loss absolute for the matrix2D (Y is a matrix 2D)
def get_loss_matrix2d(Ypre, Y):
        Ypre = np.reshape(Ypre, [-1])
        Y = np.reshape(Y, [-1])
        resul = np.sum(np.absolute(Ypre - Y)) 
        si = len(Y)
        resul = resul / si
        return resul

# get loss absolute general  (Y is a matrix 2D or a vector)    
def get_loss(Ypre, Y):
    size_Y = np.size(Y)
    size_Y0 = np.size(Y, 0)
    if (size_Y == size_Y0) :
        return get_loss_vector(Ypre, Y)
    else:
        return get_loss_matrix2d(Ypre, Y)
    
# get loss MSE for the vectors (Y is a list of number)
def get_loss_vector_MSE(Ypre, Y):
        resul = np.sum(np.square(Ypre - Y))
        return resul/len(Y)

# get loss MSE for the matrix2D (Y is a matrix 2D)
def get_loss_matrix2d_MSE(Ypre, Y):
        Ypre = np.reshape(Ypre, [-1])
        Y = np.reshape(Y, [-1])
        resul = np.sum(np.square(Ypre - Y)) 
        si = len(Y)
        resul = resul / si
        return resul
    
# get loss MSE general  (Y is a matrix 2D or a vector)    
def get_loss_MSE(Ypre, Y):
    size_Y = np.size(Y)
    size_Y0 = np.size(Y, 0)
    if (size_Y == size_Y0) :
        return get_loss_vector_MSE(Ypre, Y)
    else:
        return get_loss_matrix2d_MSE(Ypre, Y)

######################    
# Add in 09/04/2020
######################

# get loss MSE for the vectors (Y is a list of number)
def get_loss_vector_RSS(Ypre, Y):
        resul = np.sum(np.square(Ypre - Y))
        return resul

# get loss MSE for the matrix2D (Y is a matrix 2D)
def get_loss_matrix2d_RSS(Ypre, Y):
        Ypre = np.reshape(Ypre, [-1])
        Y = np.reshape(Y, [-1])
        resul = np.sum(np.square(Ypre - Y)) 
        
        return resul
    
# get loss MSE general  (Y is a matrix 2D or a vector)    
def get_loss_RSS(Ypre, Y):
    size_Y = np.size(Y)
    size_Y0 = np.size(Y, 0)
    if (size_Y == size_Y0) :
        return get_loss_vector_RSS(Ypre, Y)
    else:
        return get_loss_matrix2d_RSS(Ypre, Y)
    

# get loss MSE for the vectors (Y is a list of number)
def get_loss_vector_RMSE(Ypre, Y):
        resul = np.sum(np.square(Ypre - Y))
        return math.sqrt(resul/len(Y))

# get loss MSE for the matrix2D (Y is a matrix 2D)
def get_loss_matrix2d_RMSE(Ypre, Y):
        Ypre = np.reshape(Ypre, [-1])
        Y = np.reshape(Y, [-1])
        resul = np.sum(np.square(Ypre - Y)) 
        si = len(Y)
        resul = resul / si
        return math.sqrt(resul)
    
# get loss MSE general  (Y is a matrix 2D or a vector)    
def get_loss_RMSE(Ypre, Y):
    size_Y = np.size(Y)
    size_Y0 = np.size(Y, 0)
    if (size_Y == size_Y0) :
        return get_loss_vector_RMSE(Ypre, Y)
    else:
        return get_loss_matrix2d_RMSE(Ypre, Y)

    
def get_loss_vector_RSE(Ypre, Y):
        resul1 = np.sum(np.square(Ypre - Y))
        meanY= np.mean(Y)
        list_mean = []
        for i in range(len(Y)):
            list_mean.append(meanY)
            
        resul2 = np.sum(np.square(list_mean - Y))
        return resul1/resul2

# get loss MSE for the matrix2D (Y is a matrix 2D)
def get_loss_matrix2d_RSE(Ypre, Y):
        Ypre = np.reshape(Ypre, [-1])
        Y = np.reshape(Y, [-1])
        resul = get_loss_vector_RSE(Ypre, Y)
        return resul
    
# get loss MSE general  (Y is a matrix 2D or a vector)    
def get_loss_RSE(Ypre, Y):
    size_Y = np.size(Y)
    size_Y0 = np.size(Y, 0)
    if (size_Y == size_Y0) :
        return get_loss_vector_RSE(Ypre, Y)
    else:
        return get_loss_matrix2d_RSE(Ypre, Y)
    
    
#################################
#################################
# Get Matrix2d - batch_size in Training Data
def get_batch_matrix2d(index, batch_size, train_X, train_Y):
    m = train_X.shape[0]
    nX = train_X.shape[1]
    nY = train_Y.shape[1]
    matrixX = np.zeros((batch_size, nX))
    matrixY = np.zeros((batch_size, nY))
    for i in range(batch_size):
        index += 1
        if index >= m:
            index = 0
        for j in range(nX):
            matrixX[i][j] = train_X[index][j]
        for j in range(nY):
            matrixY[i][j] = train_Y[index][j]
        
    return (index, matrixX, matrixY)




'''
Implementation using scikit-learn library.

Project: Smart Village

Time series Prediction

Author:  HO Van Hieu
'''
import sys
import numpy as np

from sklearn import linear_model
from sklearn import kernel_ridge
from sklearn import gaussian_process
from sklearn import naive_bayes
from sklearn import svm
from sklearn import tree
from sklearn import ensemble
from sklearn import isotonic
from sklearn import neural_network
from sklearn import neighbors 

from smartSV import vocabulariesSV as vc 
import copy

gprkernel = gaussian_process.kernels.RBF(50., (50., 50.))
# list of methods
methods = {
    
    vc.SVR_rbf : svm.SVR(kernel = 'rbf'),
    
    vc.SVR_poly : svm.SVR(kernel = 'poly'),
    vc.SVR_poly1 : svm.SVR(kernel = 'poly', degree=1),
    vc.SVR_poly2 : svm.SVR(kernel = 'poly', degree=2),
    vc.SVR_poly3 : svm.SVR(kernel = 'poly', degree=3),
    vc.SVR_poly4 : svm.SVR(kernel = 'poly', degree=4),
    vc.SVR_poly5 : svm.SVR(kernel = 'poly', degree=5),
    
    vc.SVR_linear : svm.SVR(kernel = 'linear'),
    
    vc.SGDRegressor : linear_model.SGDRegressor(),
    vc.BayesianRidge : linear_model.BayesianRidge(),
    
    vc.Lars : linear_model.Lars(), #add in 06 May 2020
    
    vc.LassoLars : linear_model.LassoLars(),
    vc.LassoRegressor: linear_model.Lasso(), #add in 06 May 2020
    
    vc.ARDRegression : linear_model.ARDRegression(),
    vc.PassiveAggressiveRegressor : linear_model.PassiveAggressiveRegressor(),
    vc.TheilSenRegressor : linear_model.TheilSenRegressor(),
    vc.LinearRegression : linear_model.LinearRegression(),
    vc.HuberRegressor : linear_model.HuberRegressor(),
    vc.TheilSenRegressor : linear_model.TheilSenRegressor(),
    vc.RANSACRegressor : linear_model.RANSACRegressor(),
    vc.OrthogonalMatchingPursuit : linear_model.OrthogonalMatchingPursuit(),
    
    vc.RidgeRegressor: linear_model.Ridge(), #add in 06 May 2020
    vc.RidgeCVRegressor: linear_model.RidgeCV(), #add in 06 May 2020
    
    vc.ElasticNetCVRegressor: linear_model.ElasticNetCV(), #add in 06 May 2020
    vc.ElasticNetRegressor: linear_model.ElasticNet(), #add in 06 May 2020
    
       
    
    vc.KernelRidge : kernel_ridge.KernelRidge(),

    vc.GaussianProcessRegressor : gaussian_process.GaussianProcessRegressor(),
    
    vc.DecisionTreeRegressor : tree.DecisionTreeRegressor(),
    
    vc.RandomForestRegressor : ensemble.RandomForestRegressor(),
    vc.ExtraTreesRegressor : ensemble.ExtraTreesRegressor(),
    vc.GradientBoostingRegressor : ensemble.GradientBoostingRegressor(),
    #vc.VotingRegressor: ensemble.VotingRegressor(), #add in 06 May 2020
    
    
    vc.NNRegressor : neural_network.MLPRegressor(),
    vc.MLPRegressor2 : neural_network.MLPRegressor((100, 80)),
    vc.MLPRegressor3 : neural_network.MLPRegressor((100, 80, 60)),
    vc.KNNRegressor : neighbors.KNeighborsRegressor(),
    vc.KNNRegressor5 : neighbors.KNeighborsRegressor(n_neighbors=5),
    vc.KNNRegressor7 : neighbors.KNeighborsRegressor(n_neighbors=7),
    vc.KNNRegressor3 : neighbors.KNeighborsRegressor(n_neighbors=3),
    vc.KNNRegressor2 : neighbors.KNeighborsRegressor(n_neighbors=2),
    vc.KNNRegressor4 : neighbors.KNeighborsRegressor(n_neighbors=4),
    vc.KNNRegressor6 : neighbors.KNeighborsRegressor(n_neighbors=6),
    vc.GaussianProcessRegressorRBF : gaussian_process.GaussianProcessRegressor(kernel=gprkernel),
    vc.BaggingRegressor : ensemble.BaggingRegressor(),
    vc.AdaBoostRegressor : ensemble.AdaBoostRegressor()
    
    
    
    #MLPRegressor
 
}

#defini a class model
class model_scikitLearn(object):
    #To build the model when instantiated
    # method: name of method (NN2hd, LR ..?)
    # trainX: DataSet for training
    # trainY: LabelSet for training
    def __init__(self, method, trainX, trainY):
        self.method = method
        self.trainX = trainX
        self.trainY = trainY
        self.nX = trainX.shape[1]
        self.nY = trainY.shape[1]
        self.model_method = self.set_model_method()
    
    # to buil model (training phase)
    # trainX: DataSet for training - numpyarray
    # trainY: LabelSet for training - numpyarray
    def set_model_method(self):
        
        if (self.nY == 1):
            models = methods[self.method]
            models.fit(self.trainX,self.trainY)
        else:
            models = []
            for i in range(self.nY):
                modelI = methods[self.method]
                modelI = modelI.fit(self.trainX,self.trainY[:, i])
                
                models.append(copy.deepcopy(modelI))
    
        return models
    
    # get the prediction
    # Xtest: DataSet - numpyarray
    # return: the prediction - numpyarray
    def get_predict(self, Xtest):
        # nb_rowY x nb_columnY is the size of predictY
        nb_rowPre = Xtest.shape[0] 
        nb_columnPre = self.nY
        # pred_Y is used for saving the predict
        if (self.nY >1):
            pred_Y = []
            for i in range(self.nY):
                pred_Yi = self.model_method[i].predict(Xtest)
                pred_Y.append(copy.deepcopy(pred_Yi))
        else:
            pred_Y = self.model_method.predict(Xtest)
    
        # transfor list to array
        pred_Y = np.array(pred_Y)
        
        if (self.nY > 1):
            pred_Y = pred_Y.reshape((nb_columnPre, nb_rowPre))
            pred_Y = pred_Y.transpose()
        
        return pred_Y
    
    
       
    # get the prediction
    # Xtest: DataSet - numpyarray
    # return: the prediction - numpyarray
    def get_predict2(self, Xtest):
        # nb_rowY x nb_columnY is the size of predictY
        nb_rowPre = Xtest.shape[0] 
        nb_columnPre = self.nY
        # pred_Y is used for saving the predict
        if (self.nY >1):
            pred_Y = []
            for i in range(self.nY):
                
                modelI = methods[self.method]
                modelI = modelI.fit(self.trainX,self.trainY[:, i])
                pred_Yi = modelI.predict(Xtest)
                pred_Y.append(pred_Yi)
        else:
            modelI = methods[self.method]
            modelI = modelI.fit(self.trainX,self.trainY)
            pred_Y = modelI.predict(Xtest)
    
        # transfor list to array
        pred_Y = np.array(pred_Y)
        
        if (self.nY > 1):
            pred_Y = pred_Y.reshape((nb_columnPre, nb_rowPre))
            pred_Y = pred_Y.transpose()
        
        return pred_Y
    


    


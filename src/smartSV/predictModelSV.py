'''
Project: Smart Village
predictModelSV Class
Author:  HO Van Hieu
'''

import copy

from smartSV import vocabulariesSV as vc
from smartSV import scikitLearnSV as sksv
#from smartSV import kerasSV as krsv
#from smartSV import tensorflowSV as tssv
#import pyTorchSV as ptsv




# list of librairies in predictModelSV
librariesSV = [
    vc.scikitLearn
#    vc.keras,
#    vc.tensorFlow
    #, vc.pyTorch
]

# define prediction model
class PredictModel(object):
    
    def __init__(self, library, method):
        """
        Create a model
        - library: library name = a string (see in vocabulariesSV.py)
        - method: method name  = a string (see in vocabulariesSV.py)
        """
        self.library = library
        self.method = method
        self.model = None
        self.predict = None

    
    def set_model(self, trainX, trainY):
        """
        To Build the model
        - trainX: numpy array  - Data set
        - trainY: numpy array  - Label set
        """
        
        if(self.library  == vc.scikitLearn):
            self.model = sksv.model_scikitLearn(self.method, trainX, trainY)
        #elif (self.library  == vc.keras):
            #self.model = krsv.model_keras(self.method, trainX, trainY)
        #elif (self.library  == vc.tensorFlow):
            #self.model = tssv.model_tensorflow(self.method, trainX, trainY)
        
        #elif (self.library  == vc.pyTorch):
        #    self.model = ptsv.model_pyTorch(self.method, trainX, trainY)
        else:
            self.model = None
        
        return self
    
    
    def get_predict(self, testX):
        """
        Produce the prediction of testX
        - testX: numpy array  - Testing Data set
        - Return: prediction set of testX after applying the predict model
        """
        if(self.library  in librariesSV):
            predict = self.model.get_predict(testX)
        
        
        #elif (self.library  == "pyTorch"):
        #    self.model = ptsv.model_pyTorch.get_predict(testY)
        else:
            predict = None
        
        
        return predict
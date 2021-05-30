'''
project: Smart Village
optimizationSV package 
Author: HO Van Hieu
'''
import numpy as np
import copy
import time

from smartSV import processDataSV as pd
from smartSV import calculatorSV as cc
from smartSV import vocabulariesSV as vc 
from scipy.optimize import minimize



class Optimization_method(object):
    """
    Optimization Method:
    from the scipy library
    - method_name: name of method (in "SLSQP", "L-BFGS-B", "TNC")
    - options(see more in documents about scipy)
    """
    # initialize model
    def __init__(self, method_name, options):
        self.method_name = method_name
        self.options = options

    def set_method(self, method_name, option):
        self.method_name = method_name
        self.option = option
        return self
    
    def get_method_name(self):
        return self.method_name
    
    def get_options(self):
        return self.options
    

class OptimizerSV(object):
    """
    OptimizerSV class
    
    - modelPredictSV: predict DER Model (DE -> R)
    - optimization_method: optimization method (with a name and options)
    - columns_DinDE: index list of columnsD in columnsDE
    - xiDE: a record DE
    - xi0: the initial value of the optimization
    - bounds_D: bounds of xi0. bounds list of DE data for the optimization
    
    """
    def __init__(self, modelPredictSV, optimization_method, columns_DinDE):
        '''
        Constructor
        '''
        self.optimization_method = optimization_method
        self.columns_DinDE = columns_DinDE
        self.modelPredictSV = modelPredictSV
        self.xiDE = None
        self.xi0 = None
        self.bounds_D = None
        self.bounds = (0.0, 1.0)
    
    # bounds: define bounds of Decision values 
    def set_Bounds(self, bounds):
        self.bounds = bounds
    
    # options_op: optimisation options
    def set_optimization_method(self, optimization_method):
        self.optimization_method = optimization_method
        
    
    # set bounds list of DE data for the optimisation
    # if index in D: set the bounds for value
    #    else: set the Environment-value for value corespond
    def set_bounds_D(self):
        boundsL = []
        nb_columns = len(self.xi0)
        for i in range(nb_columns):
            if (i in self.columns_DinDE):
                boundsL.append(self.bounds)
            else:
                boundsL.append((self.xi0[i], self.xi0[i]))
        self.bounds_D = boundsL
        
    # set the first value for the optimisation
    def set_xi0(self):
        bnds_mean = (self.bounds[0] + self.bounds[1])/2
        bnds_mean = 0.5
        #bnds_mean = 0
        x0 = []
        nb_columns = len(self.xiDE)
        for i in range(nb_columns):
            if (i in self.columns_DinDE):
                x0.append(bnds_mean)
            else:
                x0.append(self.xiDE[i])
        self.xi0 = x0
    
    # set valur of xiDE. -> set value of xi0 and  Bounds_Dc.
    def set_xiDE(self, xiDE):
        self.xiDE = xiDE
        self.set_xi0()
        self.set_bounds_D()
    
    # Function for the omtimisation (min(xxx-value) -> max)
    def op_function(self, x):
        xx = np.matrix([x])
        try:
            predict = self.modelPredictSV.get_predict(xx)
        except:
            predict = self.modelPredictSV.get_model().get_predict(xx)
        predict = np.reshape(predict, (-1))
        
        return (1.0 - predict[0])
    
   
    
    # Get Res value of the optimation
    def getRes(self):
    
        res_xDE = minimize(self.op_function, self.xi0,
                           method=self.optimization_method.get_method_name(),
                           bounds=self.bounds_D,
                           options=self.optimization_method.get_options())
        
        return res_xDE
    

#################################
#Fonctions
#################################

def get_listColumn_D_in_DE(columns_D, columns_DE):
    """    
    get indexs of D in DE
    Ex: D = [1, 3, 4]; E = [1, 2, 3, 4, 5, 6] => DinE= [0, 2, 3]
    """
    columns_D_in_DE = []
    for item in columns_D:
        index = columns_DE.index(item)
        columns_D_in_DE.append(index)
    return columns_D_in_DE
    
        


def produce_xDE(columnsDE, columnsE, xE):
    
    """
    Produce xDE from xE. D_values are fixed by 0.5
    - columnsDE: index list of DE in DER data
    - columnsE: index list of E in DER data
    - xE: a record of E-Data
    """
    
    product_xDE = []
    index = 0
    for i in columnsDE:
        if (i in columnsE):
            product_xDE.append(xE[index])
            index +=1
        else:
            product_xDE.append(0.5)
    
    return product_xDE



def get_result_op_DER_once(bestModel_DER, optimization_method, columns_DinDE, xiDE):
    """
    Optimize a record xiDE
    - bestModel_DER: Machine LEarning Model for forecasting DE-> R
    - optimization_method: optimisation method (Optimization_method object)
    - columns_DinDE: indexs of D in DE
    - xiDE: a record of xiDE value
    - return:
        + result_op: Result value(s) of optimisation
        + result_pr: Result value(s) of prediction 
        + rsX: DE value after optimisation step
        + opSV.xi0: DE value before optimisation step
    """
    opSV = OptimizerSV(bestModel_DER, optimization_method, columns_DinDE)
    opSV.set_xiDE(xiDE)
    res = opSV.getRes()
    result_op = opSV.modelPredictSV.get_predict(np.array([res.x]))
    result_pr = opSV.modelPredictSV.get_predict(np.array([opSV.xiDE]))
    if(result_op < result_pr):
        rsX = opSV.xiDE
        rsOp = result_pr
    else:
        rsX = res.x
        rsOp = result_op
    
    return rsOp, result_pr, rsX, opSV.xi0



def get_result_op_DER(bestModel_DER, optimization_method, columns_DinDE, XiDE):
    
    """
    Optimize a array XiDE
    - bestModel_DER: Machine LEarning Model for forecasting DE-> R
    - optimization_method: optimisation method (Optimization_method object)
    - columns_DinDE: indexs of D in DE
    - XiDE: an array of XiDE values
    - return:
        + result_op_list: list of Result value(s) of the optimisation
        + renx_list: list of DE value after the optimisation step
    """
    
    opSV = OptimizerSV(bestModel_DER, optimization_method, columns_DinDE)
    
    
    result_op_list = []
    result_pr_list = []
    renx_list = []
    for xiDE in XiDE:
        opSV.set_xiDE(xiDE)
        res = opSV.getRes()
        result_op = opSV.modelPredictSV.get_predict(np.array([res.x]))
        result_pr = opSV.modelPredictSV.get_predict(np.array([opSV.xiDE]))
        if(result_op < result_pr):
            rsX = opSV.xiDE
            rsOp = result_pr
        else:
            rsX = res.x
            rsOp = result_op
        
        
        result_op_list.append(rsOp[0])
        renx_list.append(rsX)
    
    return result_op_list, renx_list



   
def loss_opp(type_minmax, dest_value, opRLs):
    """
    The average loss between a list of opRLs and dest_value:
    - dest_value: the value we want to attain
    - opRLs: R-optimisation value
        if type_minmax = "max"
                then sumloss += dest_value - opR
        if type_minmax = "in"
                then sumloss += opR - dest_value
    
    """
    
    sumloss = 0.0
    if(type_minmax == "max"):
        for opR in opRLs:
            if opR<dest_value:
                sumloss += dest_value - opR
    else:
        for opR in opRLs:
            if opR>dest_value:
                sumloss += opR - dest_value
                
    return sumloss/(len(opRLs))


def ex_optimization(bestModel_DER, optimization_method, columns_DinDE, XiDE):
    
    """
    Execute the optimization
    - bestModel_DER: Machine LEarning Model for forecasting DE-> R
    - optimization_method: optimisation method (Optimization_method object)
    - columns_DinDE: indexs of D in DE
    - XiDE: an array of xiDE values
    - return:
        + opRL: list of Result value(s) of the optimisation
        + renXs: list of DE value after the optimisation step
    
    """
    
    try:
        opRL, renXs= get_result_op_DER(bestModel_DER.get_model(),
                                       optimization_method,
                                       columns_DinDE,
                                       XiDE)
    except:
        opRL, renXs= get_result_op_DER(bestModel_DER,
                                       optimization_method,
                                       columns_DinDE,
                                       XiDE)
       
    opRL = np.array(opRL)
    return opRL, renXs



def get_model_ops(method_op_names = vc.method_op_names,
                  maxiter = vc.nb_maxiter,
                  ftols = vc.option_op_ftols,
                  gtols = vc.option_op_gtols,
                  eps = vc.option_op_eps):
    """
    Produce a list of optimization methods 
    - method_op_names: list of ompimization method names
    - {maxiter,ftols,tols,eps}: Optimization method option
                (see in document of scipy to know more about these options)
    """
    model_ops =[]
    for method in method_op_names:
        if (method == "SLSQP"):
            for ftol in ftols:
                for ep in eps:
                    option_op = {'maxiter': maxiter, 'ftol': ftol,
                                     'eps': ep,'disp': False}
                    optimization_method = Optimization_method(method, option_op)
                    model_ops.append(optimization_method)
                        
        else:
            for ftol in ftols:
                for ep in eps:
                    for gtol in gtols:
                        option_op = {'maxiter': maxiter, 'ftol': ftol,'gtol': gtol,
                                         'eps': ep,'disp': False}
                        optimization_method = Optimization_method(method, option_op)
                        model_ops.append(optimization_method)
                
    return model_ops
    
    
   
    
def get_best_model_op(bestModel_DER, columns_DinDE, XiDEs):
    """
    Find the best optimization model
    - bestModel_DER: Machine LEarning Model for forecasting DE-> R
    - columns_DinDE: indexs of D in DE
    - XiDEs: an array of XiDE values
    
    """
    opRL = None
    best_model_op = None
    min_loss = 1000
    model_ops = get_model_ops()
    for model_op in model_ops:
        opRL, renXs = ex_optimization(bestModel_DER,
                                      model_op,
                                      columns_DinDE,
                                      XiDEs)
        
        opRL = np.array(opRL).reshape(-1)
        loss_op = loss_opp("max", 1, opRL)
        #print(loss_op)
        if(min_loss>loss_op):
            min_loss = loss_op
            best_model_op = model_op
            #print("*", min_loss)
        
    return best_model_op, min_loss
    


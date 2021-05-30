

import sys
sys.path.insert(0, 'smartSV')
import numpy
from smartSV import processDataSV as pd
from smartSV import scikitLearnSV as sk

from smartSV import modelML_methodML_TS as ml_TS

class TS_predict:
    "This is a person class"
    def __init__(self,
                 df,
                 long_predict=7,
                 nb_previous=[7],
                 algos=['LinearRegression','HuberRegressor']

                ):

        self.df=df
        self.long_predict = long_predict
        self.nb_previous = nb_previous
        self.algos = algos
        self.algosl = [["scikitLearn", item] for item in self.algos]
        self.bestmodel = self.get_bestModel()

    def get_bestModel(self):
        methodML_TSs = ml_TS.get_methodML_TSs(self.nb_previous,
                                              [self.long_predict],
                                              self.algosl)

        matrix = self.df.values

        # define parameters about the forecasting
        columnsX = [1]
        columnsY = [1]

        # product trainning set and testing set
        percentTrain = 80

        # get the best model and its loss
        best_model, best_loss = ml_TS.get_best_model_TS_from_matrix(matrix,
                                                                    columnsX,
                                                                    columnsY,
                                                                    methodML_TSs,
                                                                    percentTrain)

        print("best model: previous_days, next_days, librairy, method_name:")
        print(best_model.get_methodML().get_infos())
        print("with loss MSE = ", best_loss)

        return best_model.get_methodML()

    def get_predict(self):
        columns_X = [1]
        columns_Y = [1]

        X, Y = pd.productXY_timeseries_from_matrix(self.df.values,
                                                   columns_X,
                                                   columns_Y,
                                                   self.bestmodel.get_nb_previousDays(),
                                                   self.long_predict)




        model = sk.model_scikitLearn(self.bestmodel.get_methodName(), X, Y)


        #testX = (self.df.values[self.df.columns[1]])[:self.bestmodel.get_nb_previousDays()]
        testX =self.df[self.df.columns[1]]
        testX =[testX[:self.bestmodel.get_nb_previousDays()]]
        testX = numpy.array(testX)

        predict = model.get_predict(testX)

        return predict[0]

class TSs_predict_multi:
    "This is a person class"
    def __init__(self,
                 df01,
                 dfinfos,
                 columnsX,
                 columnY,
                 long_predict,
                 algos,
                 nb_previous,
                 percentTrain
                 ):

        self.df01=df01
        self.long_predict = long_predict
        self.nb_previous = nb_previous
        self.algos = algos
        self.dfinfos =dfinfos
        self.columnsX=columnsX
        self.columnY=columnY
        self.percentTrain= percentTrain
        self.algosl = [["scikitLearn", item] for item in self.algos]
        self.bestModel = self.get_bestModel()
        self.predicted = self.get_predict()

    def get_bestModel(self):
        methodML_TSs = ml_TS.get_methodML_TSs(self.nb_previous,
                                              [self.long_predict],
                                              self.algosl)

        matrix = self.df01.values



        # get the best model and its loss
        best_model, best_loss = ml_TS.get_best_model_TS_from_matrix(matrix,
                                                                    self.columnsX,
                                                                    [self.columnY],
                                                                    methodML_TSs,
                                                                    self.percentTrain)

        print("best model: previous_days, next_days, librairy, method_name:")
        print(best_model.get_methodML().get_infos())
        print("with loss MSE = ", best_loss)

        return best_model.get_methodML()

    def get_predict(self):
        X, Y = pd.productXY_timeseries_from_matrix(self.df01.values,
                                                   self.columnsX,
                                                   [self.columnY],
                                                   self.bestModel.get_nb_previousDays(),
                                                   self.long_predict)

        model = sk.model_scikitLearn(self.bestModel.get_methodName(), X, Y)

        testX = []
        previous=self.bestModel.get_nb_previousDays()
        df1X=self.df01[-previous:]
        for i in range(previous):
            for j in self.columnsX:
                testX.append(df1X.values[i][j])

        testX =numpy.array([testX])
        predict = model.get_predict(testX)

        return predict[0]

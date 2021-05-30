'''
project: Smart Village
class Data_processing package
Author: HO Van Hieu
'''

from __future__ import print_function
import numpy as np
from sklearn import linear_model
import csv
import pandas as pd
from pandas import DataFrame
import os
import copy
import numbers



# read a matrix 2D in data file 
# listColumns: A vector of Integer which indicate the columns getted in the table returned 

def fileToListFromColumns(file_path, listColumns, delimit):
    
    Array = []
    with open(file_path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=delimit)
        for row in readCSV:
            ar = []
            lenListColumns = len(listColumns)
            for i in range(lenListColumns):
                ar.append(row[listColumns[i]])
            Array.append(ar)
            
    Array = np.array(Array)
    return Array





def read_csv(file_path, delimiter = ";", begining_line_number = 1):
    """
    Read a csv file -> return a dataframe (pandas)
    - file_path: path of source file 
    - delimiter: delimit for each cell in file data (";" or "," in general)
    - begining_line_number: line number to start 
        (index of dataframe is found from begining_line_number)
    - Return:
            + dataframe in data file
    """ 
    tamp_path = file_path+"tamps"
    ftamp = open(tamp_path, 'w')
    with open(file_path) as fp:
        line = fp.readline()
        cnt = 1
        while line:
            if cnt < begining_line_number:
                line = fp.readline()
                cnt += 1
            else:
                ftamp.write(line)
                line = fp.readline()
                
    ftamp.close()
    fp.close()
    dta = pd.read_csv(tamp_path, delimiter=';')
    return dta


def write_csv(matrix, index, file_path, delimiter):
    """
    Write a matrix2D into csv file
    - matrix: value matrix2D
    - index: list of index
    - file_path: path of source file 
    - delimiter: delimit for each cell in file (";" or "," in general)
    """ 
    df = DataFrame(matrix, columns= index)
    df.to_csv(file_path, index = None, header=True, sep = delimiter)
    
    return df
    


def productXY_timeseries_from_matrix( matrix,
                                      columns_examples,
                                      columns_lables,
                                      nbdates_data,
                                      nbdates_predict):
    
        """
        Produce Xset ane Yset from a matrix
        - columns_examples: A vector of Integer that indicate the columns in the table of examples
        - columns_lables: A vector of Integer that indicate the columns in the table of labels
        - nbdates_data: number of date of data to predict
        - nbdates_predict: number of date of predict
        - Return:
                + X set and Y set 
        """
        # get length of two lists arrColumn_examples and arrColumn_lables
        nbClData = len(columns_examples)
        nbClResu = len(columns_lables)
        
        # set number of columns for Xset and Yset
        nbClX = nbClData * nbdates_data
        nbClY = nbClResu * nbdates_predict
        
        data = np.array(matrix)
        sumOfdateBA = nbdates_data + nbdates_predict
        
        nbRecordData = data.shape[0]
        
        nbRecordXY = nbRecordData - sumOfdateBA + 1
        
        # set size of X and Y
        X = np.zeros((nbRecordXY, nbClX))
        Y = np.zeros((nbRecordXY, nbClY))
        
        # set value of each cell in X
        for i in range(nbRecordXY):
            for j in range(nbdates_data):
                for l in range(nbClData):
                    X[i][j*nbClData + l] = data[i+j][columns_examples[l]]
    
        # set value of each cell in Y
        for k in range(nbRecordXY):
            i = k + nbdates_data
            for j in range(nbdates_predict):
                for l in range(nbClResu):
                    Y[k][j*nbClResu + l] = data[i+j][columns_lables[l]]
        
        
        
        return X, Y



def productXY_DER_from_matrix(matrix, columns_examples, columns_lables):
        """
        Produce Xset ane Yset from a matrix
        - matrix: data matrix
        - columns_examples: A vector of Integer that indicate the columns in the table of examples
        - columns_lables: A vector of Integer that indicate the columns in the table of labels
        - Return: X, Y set
        """
    
    
        # get length of two lists arrColumn_examples and arrColumn_lables
        nbClData = len(columns_examples)
        nbClResu = len(columns_lables)
                
        #read data in file text (csv) and write data in matrix data
        data = np.array(matrix)
        nbRecordData = data.shape[0]
        
        # set size of X and Y
        X = np.zeros((nbRecordData, nbClData))
        Y = np.zeros((nbRecordData, nbClResu))
        
        # set value of each cell in X
        for i in range(nbRecordData):
            for j in range(nbClData):
                X[i][j] = data[i][columns_examples[j]]
    
        # set value of each cell in X
        for i in range(nbRecordData):
            for j in range(nbClResu):
                Y[i][j] = data[i][columns_lables[j]]
        
        return X, Y    
    

def fillna_LOCF_from_index(dta, index):
    """
    fill NAN value with the Last Observation Carried Forward (LOCF)
    - dta: framedata pandas
    - index: list of index (ex: ['cl1', 'cl2'])
    """
    for item in index:
        dta[item].ffill(axis = 0,  inplace = True)
    return dta


def fillna_NOCB_from_index(dta, index):
    """
    fill NAN value with the Next Observation Carried Backward (NOCB)
    - dta: framedata pandas
    - index: list of index (ex: ['cl1', 'cl2'])
    """
    for item in index:
        dta[item].bfill(axis = 0,  inplace = True)
    return dta
    

def fillna_mean_from_index(dta, index):
    """
    fill NAN value with the mean value
    - dta: framedata pandas
    - index: list of index (ex: ['cl1', 'cl2'])
    """
    
    for item in index:
        dta[item].fillna(dta[item].mean(), axis = 0, inplace = True)
    return dta


def fillna_LOCF_from_columns(dta, columns):
    """
    fill NAN value with the Last Observation Carried Forward (LOCF)
    - dta: framedata pandas
    - columns: list of columns (ex: [1, 2])
    """
    index = get_index_from_columns_in_dataframe(dta, columns)
    dta=fillna_LOCF_from_index(dta, index)
    return dta

def fillna_NOCB_from_columns(dta, columns):
    """
    fill NAN value with the Next Observation Carried Backward (NOCB)
    - dta: framedata pandas
    - columns: list of columns (ex: [1, 2])
    """
    index = get_index_from_columns_in_dataframe(dta, columns)
    dta=fillna_NOCB_from_index(dta, index)
    return dta
    

def fillna_mean_from_columns(dta, columns):
    """
    fill NAN value with the mean value
    - dta: framedata pandas
    - columns: list of columns (ex: [1, 2])
    """
    
    index = get_index_from_columns_in_dataframe(dta, columns)
    dta=fillna_mean_from_index(dta, index)
    return dta

    

def get_index_from_columns_in_dataframe(dta, columns):
    """
    get index from a list of columns in a dataframe 
    - dta: framedata pandas
    - columns: list of columns (ex: [1, 2])
    - Return: list of name of column (ex: ["f01", "f02"])
    """
    indexdta = dta.columns
    index = []
    for item in columns:
        index.append(indexdta[item])
    return index


def get_columns_from_index_in_dataframe(dta, index):
    """
    get columns from a list of index in a dataframe 
    - dta: framedata pandas
    - index: list of index (ex: ['f1', 'f2'])
    - Return: List of columns (ex: [1, 2])
    """
    indexdta = list(dta.columns)
    columns = []
    for item in index:
        columns.append(indexdta.index(item))
    return columns
    

    
def get_matrix_from_index_in_dataframe(dta, index):
    """
    get a matrix from index list in a dataframe
    Ex: values of dataframe is a matrix 2D with 3 index in a list (['c1', 'c2', 'c3'])
        if we want to extracte a matrix 2D with 2 index in a list ('c1', 'c3')
        we can use get_matrix_from_index_in_dataframe(dta, ['c1', 'c3'])
    """
    return (dta[index]).values


    
def get_matrix_from_columns_in_dataframe(dta, colums):
    """
    get a matrix from index list in a dataframe
    Ex: values of dataframe is a matrix 2D with 3 index in a list (['c1', 'c2', 'c3'])
        if we want to extracte a matrix 2D with 2 index in a list ('c1', 'c3')
        we can use get_matrix_from_index_in_dataframe(dta, ['c1', 'c3'])
    """
    
    index = get_index_from_columns_in_dataframe(dta, colums)
    matrix = get_matrix_from_index_in_dataframe(dta, index)
    return matrix


def is_numerical_matrix(matrix):
    """
    Check a numerical matrix
    if all values in matrix are numerical,
    return True
    if not return False
    """
    rows, cls = matrix.shape
    for i in range(rows):
        for j in range(cls):
            if (not (isinstance(matrix[i][j], (int, float, complex)))):
                return False
    return True


def is_existe_nan_in_matrix(matrix):
    """
    Check a numerical matrix
    if a value in matrix is Nan value,
    return True
    if not return False (all values are not Nan value)
    """
    rows, cls = matrix.shape
    for i in range(rows):
        for j in range(cls):
            value = matrix[i][j]
            if ((value is np.nan)or pd.isnull(value)):
                return True
    return False


def searchIndex_Min_inColumn_inMatrix2D(matrix2D, index_column):
    """
    Search index_row of min value in index_column in matrix2D
    - matrix2D: matrix data (2 dimentions)
    - indexColumn: index of column in matrix
    - Return: index of Minimal value in the column "indexColumn" of matrix
    """
    elementMin = matrix2D[0][index_column]
    indexRowMin = 0
    n = len(matrix2D)
    for i in range (1, n, +1):
        if(matrix2D[i][index_column] < elementMin):
            elementMin = matrix2D[i][index_column]
            indexRowMin = i
    return indexRowMin

def searchMinColumn_inMatrix2D(matrix2D, indexColumn):
    """
    Search the Minimal value in a column of a matrix2D
    - matrix2D: matrix data (2 dimentions)
    - indexColumn: index of column in matrix
    - Return: Minimal value in the column "indexColumn" of matrix
    """
    minS = matrix2D[0][indexColumn]
    rows = matrix2D.shape[0]
    for i in range(rows):
        if (matrix2D[i][indexColumn] < minS):
            minS= matrix2D[i][indexColumn]
    return float(minS)



def searchMaxColumn_inMatrix2D(matrix2D, indexColumn):
    """
    Search the Maximal value in a column of a matrix2D
    - matrix2D: matrix data (2 dimentions)
    - indexColumn: index of column in matrix
    - Return: Maximal value in the column "indexColumn" of matrix
    """
    maxS = matrix2D[0][indexColumn]
    rows = matrix2D.shape[0]
    for i in range(rows):
        if (matrix2D[i][indexColumn] > maxS):
            maxS= matrix2D[i][indexColumn]
    return float(maxS)



def transforme_data_01(matrix2D, listColumns):
    """
    Transforme all value in listColumns of numerical matrix 2D to (0,1)
    new_value = (value - min_of_column)/(max_of_column - min_of_column)
    
    - matrix2D:  two dimensional matrix
    - listColumns: list of columns in the transformation
    Return:
        + listMinDistance: matrix2D (index in columns_list; min_of_column, max_of_column - min_of_column)
        
        + matrixForma: matrix2D with nb_columns = len(listcolumns), value is in (0, 1)
    """
    colums = len(listColumns)
    rows = matrix2D.shape[0]
    matrixForma = np.zeros((rows, colums))
    listMinDistance = [[np.nan for i in range(3)] for i in range(rows)]
    
    for i in range(colums):
        maxCi = searchMaxColumn_inMatrix2D(matrix2D, listColumns[i])
        minCi = searchMinColumn_inMatrix2D(matrix2D, listColumns[i])
        distanceMaxMin = maxCi - minCi
        
        listMinDistance[i][0] = listColumns[i]
        listMinDistance[i][1] = minCi
        listMinDistance[i][2] = distanceMaxMin

        for j in range(rows):
            matrixForma[j][i] = (matrix2D[j][listColumns[i]] - minCi)/ distanceMaxMin
   
    return listMinDistance, matrixForma


def transforme_data_01_index(matrix2D, index):
    """
    Transforme all value in listColumns of numerical matrix 2D to (0,1)
    new_value = (value - min_of_column)/(max_of_column - min_of_column)

    - matrix2D:  two dimensional matrix
    - listColumns: list of columns in the transformation
    Return:
        + listMinDistance: matrix2D (index in columns_list; min_of_column, max_of_column - min_of_column)

        + matrixForma: matrix2D with nb_columns = len(listcolumns), value is in (0, 1)
    """
    colums = len(index)
    rows = matrix2D.shape[0]
    matrixForma = np.zeros((rows, colums))
    listMinDistance = [[np.nan for i in range(3)] for i in range(rows)]

    for i in range(colums):
        maxCi = searchMaxColumn_inMatrix2D(matrix2D, i)
        minCi = searchMinColumn_inMatrix2D(matrix2D, i)
        distanceMaxMin = maxCi - minCi

        listMinDistance[i][0] = index[i]
        listMinDistance[i][1] = minCi
        listMinDistance[i][2] = distanceMaxMin

        for j in range(rows):
            matrixForma[j][i] = (matrix2D[j][i] - minCi) / distanceMaxMin

    return listMinDistance, matrixForma


#################################
### old codes
#################################

def changeValue_inColum(matrix2D, indexColumn, valueOriginal, valueChange):
    """
    Change value in cells (valueOriginal) of indexColumn with the new value valueChange
    - matrix2D : matrix 2 dimentions
    - indexColumn: index of column changed
    """
    for i in range(matrix2D.shape[0]):
        if (matrix2D[i][indexColumn] == valueOriginal):
            matrix2D[i][indexColumn] = valueChange
    return matrix2D



def meanValue_inColumn_Matrix2D(matrix2D, indexColumn, valueNA):
    """
    Find meanvalue in  the column (indexColumn). ignore the valueNA
    - matrix2D : matrix 2 dimentions
    """
    count = 0
    sumCl = 0
    for i in range(matrix2D.shape[0]):
        if (matrix2D[i][indexColumn] !=  valueNA):
            count += 1
            sumCl += matrix2D[i][indexColumn]
    return (sumCl/count)



def productParameters_fillMissing2(matrix2D, listClsData, idLabel, valueNA):
    """
    The last functions for fill missing values (valueNA) in matrix2D
       with Algrorithm Lenear Regression

    Product the parameters for the fillMissing
    - matrix2D: matrix data
    - listClsData: list of columns data (Xset)
    - idLabel: index of column label (Yset)
    - valueNA: value marked for missing
    - Return: 
            + trainX: Xset of training
            + trainY: Yset of training
            + listRowsPre: list of rows with the valueNA in colum idLabel
            + testX: Xset of producte valueMissing
    """
    
    trainX = []
    trainY = []
    listRowsPre = []
    testX = []
    
    lenClsData = len(listClsData)
    
    for i in range(matrix2D.shape[0]):
        ar = []
        for j in range(lenClsData):
            ar.append(matrix2D[i][listClsData[j]])
        if (matrix2D[i][idLabel] == valueNA):
            testX.append(ar)
            listRowsPre.append(i)
        else:
            trainX.append(ar)
            trainY.append([matrix2D[i][idLabel]])
        
    return (trainX, trainY, listRowsPre, testX)



def changeValueNA_with_valueRegressionLinear(matrix2D, listClsData, idLabel, valueNA):
    """
    Fill missing values (valueNA) in matrix2D
    - matrix2D: matrix data
    - listClsData: list of columns data (Xset)
    - idLabel: index of column label (Yset)
    - valueNA: value marked for missing
    - Return: 
        Matrix2D after changing valueNA with valueLR (for the column idLabel)
    """
    
    
    trainX, trainY, listRowsPre, testX = productParameters_fillMissing(matrix2D,
                                                              listClsData,
                                                              idLabel,
                                                              valueNA)
    
    model = linear_model.LinearRegression()
    model.fit(trainX,trainY)
    predY = model.predict(testX)
    
    
    for i in range(len(listRowsPre)):
        matrix2D[listRowsPre[i]][idLabel] = predY[i]
        
    return matrix2D


def productParameters_fillMissing(matrix2D, listClsData, idLabel, valueNA):
    """
    The last functions for fill missing values (valueNA) in matrix2D
       with Algrorithm Lenear Regression

    Product the parameters for the fillMissing
    - matrix2D: matrix data
    - listClsData: list of columns data (Xset)
    - idLabel: index of column label (Yset)
    - valueNA: value marked for missing
    - Return:
            + trainX: Xset of training
            + trainY: Yset of training
            + listRowsPre: list of rows with the valueNA in colum idLabel
            + testX: Xset of producte valueMissing
    """

    trainX = []
    trainY = []
    listRowsPre = []
    testX = []

    lenClsData = len(listClsData)

    for i in range(matrix2D.shape[0]):
        ar = []
        for j in range(lenClsData):
            ar.append(matrix2D[i][listClsData[j]])

        vly = matrix2D[i][idLabel]
        if (vly[0] == valueNA):
            testX.append(ar)
            listRowsPre.append(i)
        else:
            trainX.append(ar)
            trainY.append([vly[0]])

    return (trainX, trainY, listRowsPre, testX)


def changeNanValue_inColum(matrix2D, indexColumn, valueChange):
    """
    Change value in cells (valueOriginal) of indexColumn with the new value valueChange
    - matrix2D : matrix 2 dimentions
    - indexColumn: index of column changed
    """
    for i in range(matrix2D.shape[0]):
        vl = matrix2D[i][indexColumn]
        if ((vl is np.nan) or pd.isnull(vl)):
            matrix2D[i][indexColumn] = valueChange
    return matrix2D


def changeValueNA_with_valueMean(matrix2D, listCls, valueNA):
    
    """
    Fill missing values (valueNA) in matrix2D
    - matrix2D: matrix data
    - listCls: list of columns that we want to fill missing values
    - valueNA: value marked for missing
    - Return: 
            Matrix2D after changing valueNA with valueMean of its columns)
    """

    # as Type cells in A to Float32
    matrix2D = matrix2D.astype(np.float32)


    for i in listCls:
        meanCli = meanValue_inColumn_Matrix2D(matrix2D, i, valueNA)
        matrix2D = changeValue_inColum(matrix2D, i, valueNA, meanCli)
    
    return matrix2D






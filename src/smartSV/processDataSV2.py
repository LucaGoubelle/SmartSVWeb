'''
project: Smart Village
class Data_processing
Author: HO Van Hieu
'''

from __future__ import print_function
import numpy as np
from sklearn import linear_model
from pandas import DataFrame
import csv

"""
#Produce Xset ane Yset from a data file (csv file)
# datafile_path: A string representing the file path.
# delimit: delimit for each cell in file data (";" or "," in general)
# listColumn_examples: A vector of Integer that indicate the columns in the table of examples
# listColumn_lables: A vector of Integer that indicate the columns in the table of labels
# nbdates_data: number of date of data to predict
# nbdates_predict: number of date of predict
def productXYFromDataFile(datafile_path, delimit, listColumn_examples, listColumn_lables,
                 nbdates_data, nbdates_predict):
        # get length of two lists arrColumn_examples and arrColumn_lables
        nbClData = len(listColumn_examples)
        nbClResu = len(listColumn_lables)
        
        # set number of columns for Xset and Yset
        nbClX = nbClData * nbdates_data
        nbClY = nbClResu * nbdates_predict
        
        #read data in file text (csv) and write data in matrix data
        data = fileToList(datafile_path, delimit)
        data = np.array(data)
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
                    X[i][j*nbClData + l] = data[i+j][listColumn_examples[l]]
    
        # set value of each cell in Y
        for k in range(nbRecordXY):
            i = k + nbdates_data
            for j in range(nbdates_predict):
                for l in range(nbClResu):
                    Y[k][j*nbClResu + l] = data[i+j][listColumn_lables[l]]
        
        
        
        return X, Y
    
"""  

#Produce Xset ane Yset from a data file (csv file)
# datafile_path: A string representing the file path.
# delimit: delimit for each cell in file data (";" or "," in general)
# listColumn_examples: A vector of Integer that indicate the columns in the table of examples
# listColumn_lables: A vector of Integer that indicate the columns in the table of labels
# nbdates_data: number of date of data to predict
# nbdates_predict: number of date of predict
def productXY_timeseries_FromDataFile(datafile_path, delimit, listColumn_examples, listColumn_lables,nbdates_data, nbdates_predict):
        # get length of two lists arrColumn_examples and arrColumn_lables
        nbClData = len(listColumn_examples)
        nbClResu = len(listColumn_lables)
        
        # set number of columns for Xset and Yset
        nbClX = nbClData * nbdates_data
        nbClY = nbClResu * nbdates_predict
        
        #read data in file text (csv) and write data in matrix data
        data = fileToList(datafile_path, delimit)
        data = np.array(data)
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
                    X[i][j*nbClData + l] = data[i+j][listColumn_examples[l]]
    
        # set value of each cell in Y
        for k in range(nbRecordXY):
            i = k + nbdates_data
            for j in range(nbdates_predict):
                for l in range(nbClResu):
                    Y[k][j*nbClResu + l] = data[i+j][listColumn_lables[l]]
        
        
        
        return X, Y
    
"""    
#Produce Xset ane Yset from a data file (csv file)
# datafile_path: A string representing the file path.
# delimit: delimit for each cell in file data (";" or "," in general)
# listColumn_examples: A vector of Integer that indicate the columns in the table of examples
# listColumn_lables: A vector of Integer that indicate the columns in the table of labels
def productXYFromDataFile_row(datafile_path, delimit, listColumn_examples, listColumn_lables):
        # get length of two lists arrColumn_examples and arrColumn_lables
        nbClData = len(listColumn_examples)
        nbClResu = len(listColumn_lables)
                
        #read data in file text (csv) and write data in matrix data
        data = fileToList(datafile_path, delimit)
        data = np.array(data)
        nbRecordData = data.shape[0]
        
        # set size of X and Y
        X = np.zeros((nbRecordData, nbClData))
        Y = np.zeros((nbRecordData, nbClResu))
        
        # set value of each cell in X
        for i in range(nbRecordData):
            for j in range(nbClData):
                X[i][j] = data[i][listColumn_examples[j]]
    
        # set value of each cell in X
        for i in range(nbRecordData):
            for j in range(nbClResu):
                Y[i][j] = data[i][listColumn_lables[j]]
        
        return X, Y
    
"""
#Produce Xset ane Yset from a data file (csv file)
# datafile_path: A string representing the file path.
# delimit: delimit for each cell in file data (";" or "," in general)
# listColumn_examples: A vector of Integer that indicate the columns in the table of examples
# listColumn_lables: A vector of Integer that indicate the columns in the table of labels
def productXY_EDR_FromDataFile(datafile_path, delimit, listColumn_examples, listColumn_lables):
        # get length of two lists arrColumn_examples and arrColumn_lables
        nbClData = len(listColumn_examples)
        nbClResu = len(listColumn_lables)
                
        #read data in file text (csv) and write data in matrix data
        data = fileToList(datafile_path, delimit)
        data = np.array(data)
        nbRecordData = data.shape[0]
        
        # set size of X and Y
        X = np.zeros((nbRecordData, nbClData))
        Y = np.zeros((nbRecordData, nbClResu))
        
        # set value of each cell in X
        for i in range(nbRecordData):
            for j in range(nbClData):
                X[i][j] = data[i][listColumn_examples[j]]
    
        # set value of each cell in X
        for i in range(nbRecordData):
            for j in range(nbClResu):
                Y[i][j] = data[i][listColumn_lables[j]]
        
        return X, Y
    
    
# read X set and Y set in data file:
# X in file "examples_name", Y in file "lables_name"
def readXYFromFiles(dossier_path, examples_name, lables_name, delimit):
        Xfile_Path = dossier_path + "/" + examples_name
        X = np.genfromtxt(Xfile_Path, delimiter=delimit)
        Yfile_Path = dossier_path + "/" + lables_name
        Y = np.genfromtxt(Yfile_Path, delimiter=delimit)
        return X, Y

# Write Xset and Yset in data file:
# X in file "examples_name", Y in file "lables_name"    
def writeXYToFiles(X, Y, dossier_path, examples_name, lables_name, delimit):
        Xfile_Path = dossier_path + "/" + examples_name
        np.savetxt(Xfile_Path, X, delimiter=delimit)
        Yfile_Path = dossier_path + "/" + lables_name
        np.savetxt(Yfile_Path, Y, delimiter=delimit)
        return True

# read a matrix 2D in data file
# nbcolums: number of columns in file data    
def fileToList(file_path, delimit):
    
    Array = []
    with open(file_path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=delimit)
        row1 = next(readCSV)
        nbcolums = len(row1)
        ar = []
        for i in range(nbcolums):
                ar.append(row1[i])
        Array.append(ar)
        
        for row in readCSV:
            ar = []
            for i in range(nbcolums):
                ar.append(row[i])
            Array.append(ar)

    Array = np.array(Array)
    return Array

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

# Search index_row of min value in index_column in matrix2D
# index_column: index of column search
def searchIndex_Min_inColumn_inMatrix2D(matrix2D, index_column):
    elementMin = matrix2D[0][index_column]
    indexRowMin = 0
    n = len(matrix2D)
    for i in range (1, n, +1):
        if(matrix2D[i][index_column] < elementMin):
            elementMin = matrix2D[i][index_column]
            indexRowMin = i
    return indexRowMin

# get number of records in file data
def get_numberRecords_fromFileData(fileData_path, delimit):
    data = np.genfromtxt(fileData_path, delimiter=delimit)
    nb_record = len(data)
    return nb_record 

# Get the first word in a string
def firstWord(strs, delimit):
    index = strs.find(delimit)
    firstW = strs[0:index]
    return firstW


# Search  min value in index_column in matrix2D
# index_column: index of column search
def searchMin_inColumn_inMatrix2D(matrix2D, indexColumn):
    minS = matrix2D[0][indexColumn]
    rows = matrix2D.shape[0]
    for i in range(rows):
        if (matrix2D[i][indexColumn] < minS):
            minS= matrix2D[i][indexColumn]
    return minS

# Search  min value in index_column in matrix2D
# index_column: index of column search
def searchMax_inColumn_inMatrix2D(matrix2D, indexColumn):
    maxS = matrix2D[0][indexColumn]
    rows = matrix2D.shape[0]
    for i in range(rows):
        if (matrix2D[i][indexColumn] > maxS):
            maxS= matrix2D[i][indexColumn]
    return maxS

# formaliser a matrix2D.
# return1: list of distance Min-Max of each column
# return2 matrixForma (a matrix2D), value of all cells = [0; 1]
# matrix2D: matrix data
# listColumns: list of columns formalised
def formalise_Data_inMatrix2D(matrix2D, listColumns):
    colums = len(listColumns)
    rows = matrix2D.shape[0]
    matrixForma = np.zeros((rows, colums))
    listMinDistance = np.zeros((colums, 3))
    
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

# change value in cells (valueOriginal) of indexColumn with the new value valueChange
# matrix2D : matrix 2 dimentions
# indexColumn: index of column changed
def changeValue_inColum(matrix2D, indexColumn, valueOriginal, valueChange):
    for i in range(matrix2D.shape[0]):
        if (matrix2D[i][indexColumn] == valueOriginal):
            matrix2D[i][indexColumn] = valueChange
    return matrix2D


# Find meanvalue in  the column (indexColumn). ignore the valueNA
# matrix2D : matrix data,  2 dimentions
def meanValue_inColumn_Matrix2D(matrix2D, indexColumn, valueNA):
    count = 0
    sumCl = 0
    for i in range(matrix2D.shape[0]):
        if (matrix2D[i][indexColumn] !=  valueNA):
            count += 1
            sumCl += matrix2D[i][indexColumn]
    return (sumCl/count)


# the last functions for fill missing values (valueNA) in matrix2D
# with Algrorithm Lenear Regression

# product the parameters for the fillMissing
# matrix2D: matrix data
# listClsData: list of columns data (Xset)
# idLabel: index of column label (Yset)
# valueNA: value marked for missing
# Return: 
#       + trainX: Xset of training
#       + trainY: Yset of training
#       + listRowsPre: list of rows with the valueNA in colum idLabel
#       + testX: Xset of producte valueMissing
def productParameters_fillMissing(matrix2D, listClsData, idLabel, valueNA):
    
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


# fill missing values (valueNA) in matrix2D
# matrix2D: matrix data
# listClsData: list of columns data (Xset)
# idLabel: index of column label (Yset)
# valueNA: value marked for missing
# Return: 
#       Matrix2D after changing valueNA with valueLR (for the column idLabel)
def changeValueNA_with_valueRegressionLinear(matrix2D, listClsData, idLabel, valueNA):
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

def changeValueNA_with_valueMean(matrix2D, listCls, valueNA):

    # as Type cells in A to Float32
    matrix2D = matrix2D.astype(np.float32)
    for i in listCls:
        meanCli = meanValue_inColumn_Matrix2D(matrix2D, i, valueNA)
        matrix2D = changeValue_inColum(matrix2D, i, valueNA, meanCli)
    
    return matrix2D



def searchMinColumn_inMatrix2D(matrix2D, indexColumn):
    minS = matrix2D[0][indexColumn]
    rows = matrix2D.shape[0]
    for i in range(rows):
        if (matrix2D[i][indexColumn] < minS):
            minS= matrix2D[i][indexColumn]
    return minS

def searchMaxColumn_inMatrix2D(matrix2D, indexColumn):
    maxS = matrix2D[0][indexColumn]
    rows = matrix2D.shape[0]
    for i in range(rows):
        if (matrix2D[i][indexColumn] > maxS):
            maxS= matrix2D[i][indexColumn]
    return maxS

def transforme_Data_inMatrix2D(matrix2D, listColumns):
    colums = len(listColumns)
    rows = matrix2D.shape[0]
    matrixForma = np.zeros((rows, colums))
    listMinDistance = np.zeros((colums, 3))
    
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


##################################
#Ajouter 04 Fevrier 2020
#################################
class singleTimeserie(object):
    """
    
    """
  
    def __init__(self, id_column, name, data):
        self.id_column = id_column
        self.name = name
        self.data = data
    
    def get_id_column(self):
        return self.id_column
    def get_name(self):
        return self.name
    def get_data(self):
        return self.data
    
    def set_id_column(self, name):
        self.name = name
        
    def set_id_column(self, id_column):
        self.id_column = id_column
        
    def set_id_column(self, data):
        self.data = data
        
    def get_max(self):
        return max(self.data)
    
    def get_min(self):
        return min(self.min)
    
    
    def count_missing_values(self):
        count=0
        for value in self.data:
            if ((value is np.nan)or pd.isnull(value)):
                count +=1
        return count
    
    def is_numerical_column(self):
        for value in self.data:
            if (not (isinstance(value, (int, float, complex)))):
                return False
        return True
        
    
    
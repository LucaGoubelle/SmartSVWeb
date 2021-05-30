#coding:utf-8
from smartSV import processDataSV as pd
import os
from os import path
import sys
import numpy as np
from pandas import DataFrame
import copy

# input look like: python fillNanLinearRegress.py -filename --columns ---columnInvolved

def getColumnsList(s):
    return s.split(',')

if(os.path.isfile(path.abspath("data/filled/"+sys.argv[1]))):
    if sys.argv[1] != None and sys.argv[2] != None and sys.argv[3] != None:
        df = pd.read_csv("data/filled/"+sys.argv[1])
        list_ncs= []
        for item in df.columns:
            #if pd.is_numerical_matrix(np.array([df[item]])):
            list_ncs.append(item)
        
        columnsToFill = getColumnsList(sys.argv[2])
        columnInvolved = [sys.argv[3]]

        print(columnsToFill)
        print(columnInvolved)
        x = pd.get_columns_from_index_in_dataframe(df, columnsToFill)
        y = pd.get_columns_from_index_in_dataframe(df, columnInvolved)
        
        matrix2D = np.array(df.values)
        valueNan = -12345678.9
        matrix = pd.changeNanValue_inColum(matrix2D, y[0], valueNan)
        
        matrix = pd.changeValueNA_with_valueRegressionLinear(matrix2D, x, y, valueNan)
        dff = DataFrame(data=matrix, columns=list_ncs)

        pd.write_csv(dff,list_ncs,"data/filled/"+sys.argv[1],";")
else:
    if sys.argv[1] != None and sys.argv[2] != None and sys.argv[3] != None:
        df = pd.read_csv("data/dp_uploaded/"+sys.argv[1])
        list_ncs= []
        for item in df.columns:
            #if pd.is_numerical_matrix(np.array([df[item]])):
            list_ncs.append(item)
        
        columnsToFill = getColumnsList(sys.argv[2])
        columnInvolved = [sys.argv[3]]

        print(columnsToFill)
        print(columnInvolved)
        x = pd.get_columns_from_index_in_dataframe(df, columnsToFill)
        y = pd.get_columns_from_index_in_dataframe(df, columnInvolved)
        
        matrix2D = np.array(df.values)
        valueNan = -12345678.9
        matrix = pd.changeNanValue_inColum(matrix2D, y[0], valueNan)
        
        matrix = pd.changeValueNA_with_valueRegressionLinear(matrix2D, x, y, valueNan)
        dff = DataFrame(data=matrix, columns=list_ncs)

        pd.write_csv(dff,list_ncs,"data/filled/"+sys.argv[1],";")

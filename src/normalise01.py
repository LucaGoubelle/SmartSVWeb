#coding:utf-8
from smartSV import processDataSV as pd
import sys
import os
from os import path
import numpy as np
from pandas import DataFrame

# python normalise01.py filePath

def re_normalise(info_form, numericalM2D_m01):
    rows, cls = numericalM2D_m01.shape
    for j in range(cls):
        minj= info_form[j][1]
        distanj = info_form[j][2]
        print("min , distance = ", minj, distanj)
        for i in range(rows):
            vl = numericalM2D_m01[i][j]
            numericalM2D_m01[i][j] = (distanj * vl) + minj

    return numericalM2D_m01

def getFileNameInFilePath(fp):
    lstfp = fp.split('/')
    return lstfp[len(lstfp)-1]

def getNewFileName_info(fp):
    fn = list(fp)
    for i in range(4):
        fn.pop()
    fn = ''.join(str(x) for x in fn)
    print(fn)
    fn += "_info.csv"
    return fn

def getNewFileName_matrix(fp):
    fn = getFileNameInFilePath(fp)
    for i in range(4):
        fn.pop()
    fn += "_01.csv"
    return fn

def isNumericalList(lst, df):
    lst_num = []
    lst_num.append(lst[0])
    for item in df.columns:
        if pd.is_numerical_matrix(np.array([df[item]])):
            print(item)
            lst_num.append(item)
    return lst_num


# input like: python normalise01.py -filename --indexCol
if(os.path.isfile(path.abspath("data/outlierDone/"+sys.argv[1]))):
    if sys.argv[1] != None:
        df = pd.read_csv("data/outlierDone/"+sys.argv[1])
        list_ncs= []
        for item in df.columns:
            #if pd.is_numerical_matrix(np.array([df[item]])):
            list_ncs.append(item)
        list_ncs = isNumericalList(list_ncs, df)

        values = df[list_ncs]
        values = np.array(values)
        info_form, numerical_01 = pd.transforme_data_01_index(values, list_ncs)
        numerical_01 = re_normalise(info_form, numerical_01)
        df_01 = DataFrame(numerical_01, columns=list_ncs)
        #df_01.insert(0, sys.argv[2], df[sys.argv[2]])
        df_info = DataFrame(info_form, columns=['index','minVL','distanceMinMax'])
        pd.write_csv(df_info, ['index','minVL','distanceMinMax'],"data/normalised01/"+getNewFileName_info(sys.argv[1]),';')
        pd.write_csv(df_01, list_ncs, "data/normalised01/"+sys.argv[1],';')

else:
    if sys.argv[1] != None:
        df = pd.read_csv("data/filled/"+sys.argv[1])
        list_ncs= []
        for item in df.columns:
            #if pd.is_numerical_matrix(np.array([df[item]])):
            list_ncs.append(item)
        list_ncs = isNumericalList(list_ncs, df)
        
        values = df[list_ncs]
        values = np.array(values)
        info_form, numerical_01 = pd.transforme_data_01_index(values, list_ncs)
        numerical_01 = re_normalise(info_form, numerical_01)
        df_01 = DataFrame(numerical_01, columns=list_ncs)
        #df_01.insert(0, sys.argv[2], df[sys.argv[2]])
        df_info = DataFrame(info_form, columns=['index', 'minVL', 'distanceMinMax'])
        pd.write_csv(df_info, ['index','minVL','distanceMinMax'],"data/normalised01/"+getNewFileName_info(sys.argv[1]),';')
        pd.write_csv(df_01, list_ncs, "data/normalised01/"+sys.argv[1],';')
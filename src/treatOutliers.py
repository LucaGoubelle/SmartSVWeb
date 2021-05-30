#coding:utf-8

#libs for Outlier methods
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from smartSV import processDataSV as pd
import numpy
import sys
import os
from os import path
from pandas import DataFrame
import numpy as np




def detect_outliers(X, method):
    model = None
    if method == "EllipticEnvelope":
        model = EllipticEnvelope(contamination=0.01)
    elif method == "IsolationForest":

        model = IsolationForest(contamination=0.1)
    elif method == "LocalOutlierFactor":
        model = LocalOutlierFactor()
    elif method == "OneClassSVM":
        model = OneClassSVM(nu=0.01)

    yhat = model.fit_predict(X)

    return yhat

def fixing_outliers(X, yhat):
    Xlist = numpy.array(X)
    lenX = len(Xlist)

    for i in range(1, lenX-1):
        if (yhat[i]==-1):
            Xlist[i]=(Xlist[i-1]+ Xlist[i+1])/2.0
    return Xlist

if(os.path.isfile(path.abspath("data/outlierDone/"+sys.argv[1]))):
	if sys.argv[1] != None and sys.argv[2] != None and sys.argv[3] == "EllipticEnvelope":
		df = pd.read_csv("data/outlierDone/"+sys.argv[1])
		list_ncs= []
		for item in df.columns:
			#if pd.is_numerical_matrix(np.array([df[item]])):
			list_ncs.append(item)
		numM2D = pd.get_matrix_from_index_in_dataframe(df, list_ncs)
		numericalM2DF = DataFrame(data=numM2D, columns=list_ncs)

		list_columns = list(numericalM2DF.columns)
		index_indfcolumns = list_columns.index(sys.argv[2])
		X = numericalM2DF.values[:, index_indfcolumns:index_indfcolumns+1]
		yhat = detect_outliers(X,sys.argv[3])
		Xlist = numpy.array(X)
		X_fix = fixing_outliers(Xlist, yhat)
		df[sys.argv[2]]= X_fix

		pd.write_csv(numericalM2DF,list_ncs,"data/outlierDone/"+sys.argv[1],";")

	elif sys.argv[1] != None and sys.argv[2] != None and sys.argv[3] == "IsolationForest":
		df = pd.read_csv("data/outlierDone/"+sys.argv[1])
		list_ncs= []
		for item in df.columns:
			#if pd.is_numerical_matrix(np.array([df[item]])):
			list_ncs.append(item)
		numM2D = pd.get_matrix_from_index_in_dataframe(df, list_ncs)
		numericalM2DF = DataFrame(data=numM2D, columns=list_ncs)

		list_columns = list(numericalM2DF.columns)
		index_indfcolumns = list_columns.index(sys.argv[2])
		X = numericalM2DF.values[:, index_indfcolumns:index_indfcolumns+1]
		yhat = detect_outliers(X,sys.argv[3])
		Xlist = numpy.array(X)
		X_fix = fixing_outliers(Xlist, yhat)
		df[sys.argv[2]]= X_fix

		pd.write_csv(numericalM2DF,list_ncs,"data/outlierDone/"+sys.argv[1],";")

	elif sys.argv[1] != None and sys.argv[2] != None and sys.argv[3] == "LocalOutlierFactor":
		df = pd.read_csv("data/outlierDone/"+sys.argv[1])
		list_ncs= []
		for item in df.columns:
			#if pd.is_numerical_matrix(np.array([df[item]])):
			list_ncs.append(item)
		numM2D = pd.get_matrix_from_index_in_dataframe(df, list_ncs)
		numericalM2DF = DataFrame(data=numM2D, columns=list_ncs)

		list_columns = list(numericalM2DF.columns)
		index_indfcolumns = list_columns.index(sys.argv[2])
		X = numericalM2DF.values[:, index_indfcolumns:index_indfcolumns+1]
		yhat = detect_outliers(X,sys.argv[3])
		Xlist = numpy.array(X)
		X_fix = fixing_outliers(Xlist, yhat)
		df[sys.argv[2]]= X_fix

		pd.write_csv(numericalM2DF,list_ncs,"data/outlierDone/"+sys.argv[1],";")

	elif sys.argv[1] != None and sys.argv[2] != None and sys.argv[3] == "OneClassSVM":
		df = pd.read_csv("data/outlierDone/"+sys.argv[1])
		list_ncs= []
		for item in df.columns:
			#if pd.is_numerical_matrix(np.array([df[item]])):
			list_ncs.append(item)
		numM2D = pd.get_matrix_from_index_in_dataframe(df, list_ncs)
		numericalM2DF = DataFrame(data=numM2D, columns=list_ncs)

		list_columns = list(numericalM2DF.columns)
		index_indfcolumns = list_columns.index(sys.argv[2])
		X = numericalM2DF.values[:, index_indfcolumns:index_indfcolumns+1]
		yhat = detect_outliers(X,sys.argv[3])
		Xlist = numpy.array(X)
		X_fix = fixing_outliers(Xlist, yhat)
		df[sys.argv[2]]= X_fix

		pd.write_csv(numericalM2DF,list_ncs,"data/outlierDone/"+sys.argv[1],";")

elif(os.path.isfile(path.abspath("data/filled/"+sys.argv[1]))):
	if sys.argv[1] != None and sys.argv[2] != None and sys.argv[3] == "EllipticEnvelope":
		df = pd.read_csv("data/filled/"+sys.argv[1])
		list_ncs= []
		for item in df.columns:
			#if pd.is_numerical_matrix(np.array([df[item]])):
			list_ncs.append(item)
		numM2D = pd.get_matrix_from_index_in_dataframe(df, list_ncs)
		numericalM2DF = DataFrame(data=numM2D, columns=list_ncs)

		list_columns = list(numericalM2DF.columns)
		index_indfcolumns = list_columns.index(sys.argv[2])
		X = numericalM2DF.values[:, index_indfcolumns:index_indfcolumns+1]
		yhat = detect_outliers(X,sys.argv[3])
		Xlist = numpy.array(X)
		X_fix = fixing_outliers(Xlist, yhat)
		df[sys.argv[2]]= X_fix

		pd.write_csv(numericalM2DF,list_ncs,"data/outlierDone/"+sys.argv[1],";")

	elif sys.argv[1] != None and sys.argv[2] != None and sys.argv[3] == "IsolationForest":
		df = pd.read_csv("data/filled/"+sys.argv[1])
		list_ncs= []
		for item in df.columns:
			#if pd.is_numerical_matrix(np.array([df[item]])):
			list_ncs.append(item)
		numM2D = pd.get_matrix_from_index_in_dataframe(df, list_ncs)
		numericalM2DF = DataFrame(data=numM2D, columns=list_ncs)

		list_columns = list(numericalM2DF.columns)
		index_indfcolumns = list_columns.index(sys.argv[2])
		X = numericalM2DF.values[:, index_indfcolumns:index_indfcolumns+1]
		yhat = detect_outliers(X,sys.argv[3])
		Xlist = numpy.array(X)
		X_fix = fixing_outliers(Xlist, yhat)
		df[sys.argv[2]]= X_fix

		pd.write_csv(numericalM2DF,list_ncs,"data/outlierDone/"+sys.argv[1],";")

	elif sys.argv[1] != None and sys.argv[2] != None and sys.argv[3] == "LocalOutlierFactor":
		df = pd.read_csv("data/filled/"+sys.argv[1])
		list_ncs= []
		for item in df.columns:
			#if pd.is_numerical_matrix(np.array([df[item]])):
			list_ncs.append(item)
		numM2D = pd.get_matrix_from_index_in_dataframe(df, list_ncs)
		numericalM2DF = DataFrame(data=numM2D, columns=list_ncs)

		list_columns = list(numericalM2DF.columns)
		index_indfcolumns = list_columns.index(sys.argv[2])
		X = numericalM2DF.values[:, index_indfcolumns:index_indfcolumns+1]
		yhat = detect_outliers(X,sys.argv[3])
		Xlist = numpy.array(X)
		X_fix = fixing_outliers(Xlist, yhat)
		df[sys.argv[2]]= X_fix

		pd.write_csv(numericalM2DF,list_ncs,"data/outlierDone/"+sys.argv[1],";")

	elif sys.argv[1] != None and sys.argv[2] != None and sys.argv[3] == "OneClassSVM":
		df = pd.read_csv("data/filled/"+sys.argv[1])
		list_ncs= []
		for item in df.columns:
			#if pd.is_numerical_matrix(np.array([df[item]])):
			list_ncs.append(item)
		numM2D = pd.get_matrix_from_index_in_dataframe(df, list_ncs)
		numericalM2DF = DataFrame(data=numM2D, columns=list_ncs)

		list_columns = list(numericalM2DF.columns)
		index_indfcolumns = list_columns.index(sys.argv[2])
		X = numericalM2DF.values[:, index_indfcolumns:index_indfcolumns+1]
		yhat = detect_outliers(X,sys.argv[3])
		Xlist = numpy.array(X)
		X_fix = fixing_outliers(Xlist, yhat)
		df[sys.argv[2]]= X_fix

		pd.write_csv(numericalM2DF,list_ncs,"data/outlierDone/"+sys.argv[1],";")

else:
	if sys.argv[1] != None and sys.argv[2] != None and sys.argv[3] == "EllipticEnvelope":
		df = pd.read_csv("data/dp_uploaded/"+sys.argv[1])
		list_ncs= []
		for item in df.columns:
			#if pd.is_numerical_matrix(np.array([df[item]])):
			list_ncs.append(item)
		numM2D = pd.get_matrix_from_index_in_dataframe(df, list_ncs)
		numericalM2DF = DataFrame(data=numM2D, columns=list_ncs)

		list_columns = list(numericalM2DF.columns)
		index_indfcolumns = list_columns.index(sys.argv[2])
		X = numericalM2DF.values[:, index_indfcolumns:index_indfcolumns+1]
		yhat = detect_outliers(X,sys.argv[3])
		Xlist = numpy.array(X)
		X_fix = fixing_outliers(Xlist, yhat)
		df[sys.argv[2]]= X_fix

		pd.write_csv(numericalM2DF,list_ncs,"data/outlierDone/"+sys.argv[1],";")

	elif sys.argv[1] != None and sys.argv[2] != None and sys.argv[3] == "IsolationForest":
		df = pd.read_csv("data/dp_uploaded/"+sys.argv[1])
		list_ncs= []
		for item in df.columns:
			#if pd.is_numerical_matrix(np.array([df[item]])):
			list_ncs.append(item)
		numM2D = pd.get_matrix_from_index_in_dataframe(df, list_ncs)
		numericalM2DF = DataFrame(data=numM2D, columns=list_ncs)

		list_columns = list(numericalM2DF.columns)
		index_indfcolumns = list_columns.index(sys.argv[2])
		X = numericalM2DF.values[:, index_indfcolumns:index_indfcolumns+1]
		yhat = detect_outliers(X,sys.argv[3])
		Xlist = numpy.array(X)
		X_fix = fixing_outliers(Xlist, yhat)
		df[sys.argv[2]]= X_fix

		pd.write_csv(numericalM2DF,list_ncs,"data/outlierDone/"+sys.argv[1],";")

	elif sys.argv[1] != None and sys.argv[2] != None and sys.argv[3] == "LocalOutlierFactor":
		df = pd.read_csv("data/dp_uploaded/"+sys.argv[1])
		list_ncs= []
		for item in df.columns:
			#if pd.is_numerical_matrix(np.array([df[item]])):
			list_ncs.append(item)
		numM2D = pd.get_matrix_from_index_in_dataframe(df, list_ncs)
		numericalM2DF = DataFrame(data=numM2D, columns=list_ncs)

		list_columns = list(numericalM2DF.columns)
		index_indfcolumns = list_columns.index(sys.argv[2])
		X = numericalM2DF.values[:, index_indfcolumns:index_indfcolumns+1]
		yhat = detect_outliers(X,sys.argv[3])
		Xlist = numpy.array(X)
		X_fix = fixing_outliers(Xlist, yhat)
		df[sys.argv[2]]= X_fix

		pd.write_csv(numericalM2DF,list_ncs,"data/outlierDone/"+sys.argv[1],";")

	elif sys.argv[1] != None and sys.argv[2] != None and sys.argv[3] == "OneClassSVM":
		df = pd.read_csv("data/dp_uploaded/"+sys.argv[1])
		list_ncs= []
		for item in df.columns:
			#if pd.is_numerical_matrix(np.array([df[item]])):
			list_ncs.append(item)
		numM2D = pd.get_matrix_from_index_in_dataframe(df, list_ncs)
		numericalM2DF = DataFrame(data=numM2D, columns=list_ncs)

		list_columns = list(numericalM2DF.columns)
		index_indfcolumns = list_columns.index(sys.argv[2])
		X = numericalM2DF.values[:, index_indfcolumns:index_indfcolumns+1]
		yhat = detect_outliers(X,sys.argv[3])
		Xlist = numpy.array(X)
		X_fix = fixing_outliers(Xlist, yhat)
		df[sys.argv[2]]= X_fix

		pd.write_csv(numericalM2DF,list_ncs,"data/outlierDone/"+sys.argv[1],";")
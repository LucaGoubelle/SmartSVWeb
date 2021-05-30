#coding:utf-8
from smartSV import processDataSV as pd
import sys
import os
from os import path
import numpy as np
from pandas import DataFrame

# input look like: python fillNan.py -filename --columns ---method

def getColumnsList(s):
	return s.split(',')

if(os.path.isfile(path.abspath("data/filled/"+sys.argv[1]))):
	if sys.argv[1] != None and sys.argv[2] != None and sys.argv[3] == "OCB":
		df = pd.read_csv("data/filled/"+sys.argv[1])
		list_ncs= []
		for item in df.columns:
			#if pd.is_numerical_matrix(np.array([df[item]])):
			list_ncs.append(item)
		numM2D = pd.get_matrix_from_index_in_dataframe(df, list_ncs)
		numericalM2DF = DataFrame(data=numM2D, columns=list_ncs)
		columnsToFill = getColumnsList(sys.argv[2])
		print(columnsToFill)
		for c in columnsToFill:
			numericalM2DF[c].bfill(axis=0, inplace=True)
			numericalM2DF[c].ffill(axis=0, inplace=True)
		pd.write_csv(numericalM2DF,list_ncs,"data/filled/"+sys.argv[1],";")

	elif sys.argv[1] != None and sys.argv[2] != None and sys.argv[3] == "LOCF":
		df = pd.read_csv("data/filled/"+sys.argv[1])
		list_ncs= []
		for item in df.columns:
			#if pd.is_numerical_matrix(np.array([df[item]])):
			list_ncs.append(item)
		numM2D = pd.get_matrix_from_index_in_dataframe(df, list_ncs)
		numericalM2DF = DataFrame(data=numM2D, columns=list_ncs)
		columnsToFill = getColumnsList(sys.argv[2])
		print(columnsToFill)
		for c in columnsToFill:
			numericalM2DF[c].ffill(axis=0, inplace=True)
			numericalM2DF[c].bfill(axis=0, inplace=True)
		pd.write_csv(numericalM2DF,list_ncs,"data/filled/"+sys.argv[1],";")

	elif sys.argv[1] != None and sys.argv[2] != None and sys.argv[3] == "mean":
		df = pd.read_csv("data/filled/"+sys.argv[1])
		list_ncs= []
		for item in df.columns:
			#if pd.is_numerical_matrix(np.array([df[item]])):
			list_ncs.append(item)
		numM2D = pd.get_matrix_from_index_in_dataframe(df, list_ncs)
		numericalM2DF = DataFrame(data=numM2D, columns=list_ncs)
		columnsToFill = getColumnsList(sys.argv[2])
		print(columnsToFill)
		for c in columnsToFill:
			numericalM2DF[c].fillna(numericalM2DF[c].mean(),axis=0, inplace=True)
		pd.write_csv(numericalM2DF,list_ncs,"data/filled/"+sys.argv[1],";")
else:
	if sys.argv[1] != None and sys.argv[2] != None and sys.argv[3] == "OCB":
		df = pd.read_csv("data/dp_uploaded/"+sys.argv[1])
		list_ncs= []
		for item in df.columns:
			#if pd.is_numerical_matrix(np.array([df[item]])):
			list_ncs.append(item)
		numM2D = pd.get_matrix_from_index_in_dataframe(df, list_ncs)
		numericalM2DF = DataFrame(data=numM2D, columns=list_ncs)
		columnsToFill = getColumnsList(sys.argv[2])
		print(columnsToFill)
		for c in columnsToFill:
			numericalM2DF[c].bfill(axis=0, inplace=True)
			numericalM2DF[c].ffill(axis=0, inplace=True)
		pd.write_csv(numericalM2DF,list_ncs,"data/filled/"+sys.argv[1],";")

	elif sys.argv[1] != None and sys.argv[2] != None and sys.argv[3] == "LOCF":
		df = pd.read_csv("data/dp_uploaded/"+sys.argv[1])
		list_ncs= []
		for item in df.columns:
			#if pd.is_numerical_matrix(np.array([df[item]])):
			list_ncs.append(item)
		numM2D = pd.get_matrix_from_index_in_dataframe(df, list_ncs)
		numericalM2DF = DataFrame(data=numM2D, columns=list_ncs)
		columnsToFill = getColumnsList(sys.argv[2])
		print(columnsToFill)
		for c in columnsToFill:
			numericalM2DF[c].ffill(axis=0, inplace=True)
			numericalM2DF[c].bfill(axis=0, inplace=True)
		pd.write_csv(numericalM2DF,list_ncs,"data/filled/"+sys.argv[1],";")

	elif sys.argv[1] != None and sys.argv[2] != None and sys.argv[3] == "mean":
		df = pd.read_csv("data/dp_uploaded/"+sys.argv[1])
		list_ncs= []
		for item in df.columns:
			#if pd.is_numerical_matrix(np.array([df[item]])):
			list_ncs.append(item)
		numM2D = pd.get_matrix_from_index_in_dataframe(df, list_ncs)
		numericalM2DF = DataFrame(data=numM2D, columns=list_ncs)
		columnsToFill = getColumnsList(sys.argv[2])
		print(columnsToFill)
		for c in columnsToFill:
			numericalM2DF[c].fillna(numericalM2DF[c].mean(),axis=0, inplace=True)
		pd.write_csv(numericalM2DF,list_ncs,"data/filled/"+sys.argv[1],";")


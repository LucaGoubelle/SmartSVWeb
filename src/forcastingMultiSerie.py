#coding:utf-8
from Mc_predictTS import TSs_predict_multi
from smartSV import processDataSV as pd
import sys

#  python forcastingMultiSerie.py -file --algos ---nbPrev ----longPred -----percTrain ------x -------y

def strtolist(s):
	return s.split(',')

def strToIntList(s):
	lst = s.split(',')
	for i in range(len(lst)):
		lst[i] = int(lst[i])
	return lst

filePath = "data/fs_uploaded/";
data_frame = pd.read_csv(filePath+sys.argv[1])

dfinfos = None
x = strtolist(sys.argv[6])
y = sys.argv[7]

x = pd.get_columns_from_index_in_dataframe(data_frame, x)
ys = pd.get_columns_from_index_in_dataframe(data_frame, [y])

lp = int(sys.argv[4])
algs = strtolist(sys.argv[2])
prev = strToIntList(sys.argv[3])
pt = int(sys.argv[5])

ts_predict = TSs_predict_multi(data_frame, dfinfos, x, ys[0], lp, algs, prev, pt)
predict = ts_predict.get_predict()
pd.write_csv(predict, ["predict"], "data/forcasted/"+sys.argv[1],";")
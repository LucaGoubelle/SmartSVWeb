#coding:utf-8
from Mc_predictTS import TS_predict
from smartSV import processDataSV as prd
import sys

def strtolist(s):
	return s.split(',')

def strToIntList(s):
	lst = s.split(',')
	for i in range(len(lst)):
		lst[i] = int(lst[i])
	return lst

filePath = "data/fs_uploaded/";
data_frame = prd.read_csv(filePath+sys.argv[1])
ts_predict = TS_predict(data_frame, long_predict = int(sys.argv[4]), nb_previous = strToIntList(sys.argv[3]), algos = strtolist(sys.argv[2]))
predict = ts_predict.get_predict()
prd.write_csv(predict, ["predict"], "data/forcasted/"+sys.argv[1],";")


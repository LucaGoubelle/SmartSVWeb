#coding:utf-8
from Mc_predictTS import TS_predict
from smartSV import processDataSV as prd
import sys
print(sys.argv[1])

filePath = "data/fs_uploaded/";
data_frame = prd.read_csv(filePath+sys.argv[1])
ts_predict = TS_predict(data_frame)
predict = ts_predict.get_predict()
prd.write_csv(predict, ["predict"], "data/forcasted/"+sys.argv[1],";")
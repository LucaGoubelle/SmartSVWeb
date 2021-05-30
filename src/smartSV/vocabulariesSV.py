

# libraries SV

scikitLearn = "scikitLearn"
keras = "keras"
tensorFlow= "tensorflow"
pyTorch = "pyTorch"

# methods in scikitLearnSV, Keras and PyTorch, Tensorflow

SVR_rbf =  'SVR_rbf'

SVR_poly =     'SVR_poly'
SVR_poly1 =     'SVR_poly1'
SVR_poly2 =     'SVR_poly2'
SVR_poly3 =     'SVR_poly3'
SVR_poly4 =     'SVR_poly4'
SVR_poly5 =     'SVR_poly5'




SVR_linear =     'SVR_linear'
SGDRegressor =     'SGDRegressor'
BayesianRidge =    'BayesianRidge'
LassoLars =    'LassoLars'
Lars = 'Lars'
LassoRegressor = 'Lasso'
RidgeRegressor = 'RidgeRegressor'
RidgeCVRegressor ='RidgeCVRegressor'
ElasticNetCVRegressor = 'ElasticNetCVRegressor'
ElasticNetRegressor = 'ElasticNetRegressor'
VotingRegressor = 'VotingRegressor'

ARDRegression =    'ARDRegression'
PassiveAggressiveRegressor =    'PassiveAggressiveRegressor'
TheilSenRegressor =    'TheilSenRegressor'
LinearRegression =    'LinearRegression'
HuberRegressor =    'HuberRegressor'
TheilSenRegressor =     'TheilSenRegressor'
RANSACRegressor =     'RANSACRegressor'
OrthogonalMatchingPursuit =    'OrthogonalMatchingPursuit'
KernelRidge =     'KernelRidge'
GaussianProcessRegressor =    'GaussianProcessRegressor'
DecisionTreeRegressor =    'DecisionTreeRegressor'
RandomForestRegressor =    'RandomForestRegressor'
ExtraTreesRegressor =    'ExtraTreesRegressor'
GradientBoostingRegressor =    'GradientBoostingRegressor'
NNRegressor =    'NNRegressor'
MLPRegressor2 = 'MLPRegressor2'
MLPRegressor3 = 'MLPRegressor3'
KNNRegressor = 'KNNRegressor'
KNNRegressor2 = 'KNNRegressor2'
KNNRegressor3 = 'KNNRegressor3'
KNNRegressor4 = 'KNNRegressor4'
KNNRegressor5 = 'KNNRegressor5'
KNNRegressor6 = 'KNNRegressor6'
KNNRegressor7 = 'KNNRegressor7'
GaussianProcessRegressorRBF = 'GaussianProcessRegressorRBF'
BaggingRegressor = 'BaggingRegressor'
AdaBoostRegressor = 'AdaBoostRegressor'




NN2hd = 'NN2hd'
CNN = 'CNN'
RNN = 'RNN'
BiRNN = 'BiRNN'

# general parameters in tha algorithms of deep learning  - Keras, Tensorflow, PyTorch
learning_rate = 0.1 # learning rate
nb_epochs = 300 # number of epochs  (20000)
batch_size = 256 # size of batch for once time. We can choose 128, 256, 512 ...
loss_type= 'mean_absolute_error' # type of loss calculation
optimizer_type= 'adam'  # type of optimisation
# the paramaters support for calculate predictSet of algrorithms in TensorflowSV
keep_prob = 0.005
nbhidden = 128



delimit = ";"

divisors = [3, 4, 5, 6, 7,10, 2]




#----------Ajuter 27 Mai 2019
# for Optimisation
range_ftols = range(1, 6)
range_gtols = range(1, 6)
range_eps = range(1, 6)
option_op_ftols = [1.0/(10.0**(2*i)) for i in range_ftols]
option_op_gtols = [1.0/(10.0**(2*i)) for i in range_gtols]
option_op_eps = [1.4901161193847656/(10.0**(2*i)) for i in range_eps]
method_op_names = ["SLSQP", "L-BFGS-B"] # , "TNC"
nb_maxiter = 10000



#****************
# for Smart systems parammeters
#****************
# definitions for Smart Entity model
statePausing = "pausing"
stateWorking = "working"
thresholdPrediction = 5

# folder where we can see data files (list of methods, data for Machine Learning...) for Smart Entity Model
folder_pathSE = "SmartEntity_dataFolder/"

# data_type (input, output data_type for each AM, CM in Smart system)
data_type_IoT_to_Epredictor = "IoT_to_Epredictor"
data_type_Epretictor_to_CMoptimizator = "Epretictor_to_CMoptimizator"
data_type_AMoptimizator_to_Dcontroler = "CMoptimizator_to_Dcontroler"
data_type_CMoptimizator_to_IoT = "CMoptimizator_to_IoT"
data_type_IoT_to_CMoptimizator = "IoT_to_CMoptimizator"



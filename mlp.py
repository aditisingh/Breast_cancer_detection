import scipy.io as sio
import numpy as np
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

Data=sio.loadmat('Data_400.mat')
train_x=Data['TrainData_400']
train_y=Data['TrainData_targets_400']
test_x=Data['TestData_400']
test_y=Data['TestData_targets_400']

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(500,),random_state=1)
clf.fit(train_x,train_y)
predict_y=clf.predict(test_x)
c=confusion_matrix(test_y,predict_y)

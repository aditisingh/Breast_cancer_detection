from __future__ import division, print_function, absolute_import
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.data_utils import shuffle, to_categorical
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split

f=sio.loadmat('features.mat')
t=sio.loadmat('targets.mat')
X=f['features']
Y=t['targets']
X=X.reshape((22143,227*227*3))
X_train,  X_test, Y_train, Y_test = train_test_split(X,Y.T,test_size=0.2,random_state=42)
X_train, Y_train = shuffle(X_train, Y_train)
Y_train = to_categorical(Y_train, 8)
Y_test = to_categorical(Y_test, 8)

encoder=tflearn.input_data(shape=[None,154587])
encoder=tflearn.fully_connected(encoder,8192)
encoder=tflearn.fully_connected(encoder,1024)
encoder=tflearn.fully_connected(encoder,256)
encoder=tflearn.fully_connected(encoder,64)
decoder=tflearn.fully_connected(encoder,256)
decoder=tflearn.fully_connected(decoder,1024)
decoder=tflearn.fully_connected(decoder,8192)
net = tflearn.regression(decoder, optimizer='adam', learning_rate=0.001,loss='mean_square', metric=None)
model = tflearn.DNN(net, tensorboard_verbose=0)


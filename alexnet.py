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
from sklearn.metrics import confusion_matrix

f=sio.loadmat('features.mat')
t=sio.loadmat('targets.mat')
X=f['features']
Y=t['targets']
X_train,  X_test, Y_train, Y_test = train_test_split(X,Y.T,test_size=0.2,random_state=42)
X_train, Y_train = shuffle(X_train, Y_train)
Y_train = to_categorical(Y_train, 8)
#Y_test = to_categorical(Y_test, 8)


# Building 'AlexNet'
network = input_data(shape=[None, 227, 227, 3])
network = conv_2d(network, 96, 11, strides=4, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 8, activation='softmax')
network = regression(network, optimizer='momentum',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Training
model = tflearn.DNN(network,tensorboard_dir='/uhpc/roysam/aditi/alexnet', checkpoint_path='/uhpc/roysam/aditi/alexnet/model_alexnet',max_checkpoints=1, tensorboard_verbose=2)
model.fit(X_train, Y_train, n_epoch=150, validation_set=0.1, shuffle=True,
          show_metric=True, batch_size=300, snapshot_step=200,
          snapshot_epoch=False, run_id='training')

y_pred=[]
for i in range(len(X_test)):
  y_pred.append(model.predict(X_test))

conf=confusion_matrix(Y_test, y_pred)
sio.savemat('confusion_matrix.mat',{'conf':conf})
  

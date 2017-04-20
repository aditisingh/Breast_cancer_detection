import os
import numpy as np
import scipy.io as sio

files=os.listdir('../features')
features=[]
targets=[]

for file in files:
  if file.endswith('.npy'):
    f=np.load(file)
    base=file[4:8]
    if(base=='B_A-'):
      class1=0
    elif(base=='B_F-'):
      class1=1
    elif(base=='B_PT'):
      class1=2
    elif(base=='B_TA'):
      class1=3
    elif(base=='M_DC'):
      class1=4
    elif(base=='M_LC'):
      class1=5
    elif(base=='M_MC'):
      class1=6
    else:
      class1=7
    targets.append(class1)
    features.append(f)

sio.savemat('features_new.mat',{'features':features})
sio.savemat('targets_new.mat',{'targets':targets})

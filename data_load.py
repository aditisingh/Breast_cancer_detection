import os
import numpy as np
from scipy.misc import imread, imresize

files=os.listdir('100_alex')
features=[]
targets=[]

for file in files:
  if file.endswith('.png'):
    name=os.path.join(100_alex/',file)
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
    img1=imread(name)
    img1=imresize(img1,(227,227))
    targets.append(class1)
    features.append(img1)

np.save('features.npy',features)
np.save('targets.npy',targets)




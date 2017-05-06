# Breast_cancer_detection
Using pre-trained model to classify images to detect cancerous cells

Pre-requirements:
1. Python2.7
2. MATLAB (LIBSVM)
3. Numpy, Scipy,Sklearn
4. Tensorflow 1.0
5. Tflearn


BreakHis dataset can be found at: http://web.inf.ufpr.br/vri/breast-cancer-database

Add all files to the same folder. Run each of them in the following order:
1. Run vgg16_cv.py to extract the features from each image of BreakHis dataset. It will create one feature file per image int he same folder
2. Run generate_features.py to combine all individual feature files into one feature matrix (mat file). It also creates a separate target mat file.
3. Run CV_balancing_code.m to treat the data imbalance. It outputs 4 files: training data, training data targets, test data and test data targets
4. Use classifier_code.m and RandomForest_CV.m to classify the data using Linear SVM, Polynomial SVM and Random Forest.
5. Run alexnet.py to get the trained AlexNet model and confusion matrix. 

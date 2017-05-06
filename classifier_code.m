
% SVM training via LIBSVM:

load CV_balanced_dataset 


model = svmtrain(balanced_TrainData_targets, balanced_TrainData, '-s 0 -t 1 -c 1 -g 2 -b 1');


[predict_label, accuracy, prob_estimates] = svmpredict(balanced_TestData_targets,balanced_TestData, model, '-b 1');

%Generating Confusion Matrix:

[Co,order] = confusionmat(balanced_TestData_targets,predict_label);
Co
order

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Random Forest training

%D=TestData_targets_100;
 %B = TreeBagger(400,TrainData_100,TrainData_targets_100);
%Yfit = predict(B,TestData_100);
%A=str2double (Yfit);
%[Co,order] = confusionmat(D,A);
%C=D-A;
 %sum(C(:)==0)
%Co
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% KNN training :

%load Data_100;
%D=TestData_targets_100;
%Class = knnclassify(TestData_100,TrainData_100,TrainData_targets_100, 20);
%[Co,order] = confusionmat(D,Class);
%%sum(C(:)==0)
%Co
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

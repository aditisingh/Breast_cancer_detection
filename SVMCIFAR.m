data_train = features(1:floor(0.8*10000),:);
target_train = targets(:,1:floor(0.8*10000));
data_test = features((floor(0.8*10000)+1):10000,:);
target_test = targets(:,(floor(0.8*10000)+1):10000); 

%Without Feature Reduction
%One vs One 
md1 = fitcecoc(data_train,target_train);
coding_matrix = md1.CodingMatrix;
insample_loss = resubLoss(md1);

cvmd1 = crossval(md1); % cross validation
outofsample_loss = kfoldLoss(cvmd1);

predicted_labels = predict(md1,data_test);
Conf_mat =confusionmat(target_test,predicted_labels);
acc_one = trace(Conf_mat)/sum(sum(Conf_mat));
precision_mat_one = precision(Conf_mat);
recall_mat_one = recall(Conf_mat);


% One vs All

md2 = fitcecoc(data_train,target_train,'coding','onevsall');
coding_matrix_all = md2.CodingMatrix;
insample_loss_all = resubLoss(md2);

cvmd2 = crossval(md2);
outofsample_loss_all = kfoldLoss(cvmd2);
predicted_labels_all = predict(md2,data_test);
Conf_mat_all =confusionmat(target_test,predicted_labels_all);
acc_all = trace(Conf_mat_all)/sum(sum(Conf_mat_all));

precision_mat_all = precision(Conf_mat_all);
recall_mat_all = recall(Conf_mat_all);

%PCA
[coeff,score,latent,tsquared,explained] = pca(data_train);
train_PCA = data_train*coeff(:,1:1500); % 1500 features cover 96.3158% of variance
test_PCA = data_test*coeff(:,1:1500);

%SVM with PCA
%one vs one

md1_PCA = fitcecoc(train_PCA,target_train);
coding_matrix_PCA = md1_PCA.CodingMatrix;
insample_loss_PCA = resubLoss(md1_PCA);

cvmd1_PCA = crossval(md1_PCA); % cross validation
outofsample_loss_PCA = kfoldLoss(cvmd1_PCA);

predict_labels_PCA = predict(md1_PCA,test_PCA);
Conf_mat_PCA =confusionmat(target_test,predict_labels_PCA);
acc_PCA = trace(Conf_mat_PCA)/sum(sum(Conf_mat_PCA));
precision_mat_PCA = precision(Conf_mat_PCA);
recall_mat_PCA = recall(Conf_mat_PCA);

%one vs all

md1_PCA_all = fitcecoc(train_PCA,target_train,'coding','onevsall');
cod_mat_PCA_all = md1_PCA_all.CodingMatrix;
insmple_lss_PCA_all = resubLoss(md1_PCA_all);

cvmd_PCA_all = crossval(md1_PCA_all); % cross validation
outofsmple_lss_PCA_all = kfoldLoss(cvmd_PCA_all);

predict_labels_PCA_all = predict(md1_PCA_all,test_PCA);
Conf_mat_PCA_all =confusionmat(target_test,predict_labels_PCA_all);
acc_PCA_all = trace(Conf_mat_PCA_all)/sum(sum(Conf_mat_PCA_all));
prec_mat_PCA_all = precision(Conf_mat_PCA_all);
recall_mat_PCA_all = recall(Conf_mat_PCA_all);

%Gaussian Kernel
%one vs one

template = templateSVM('KernelFunction','gaussian');
md_gauss = fitcecoc(train_PCA,target_train','Learners',template);
cod_mat_gauss = md_gauss.CodingMatrix;
insmple_lss_gauss = resubLoss(md_gauss);

cvmd_gauss = crossval(md_gauss);
outofsmple_lss_gauss = kfoldLoss(cvmd_gauss);
predict_labels_gauss = predict(md_gauss,test_PCA);
Conf_mat_gauss =confusionmat(target_test,predict_labels_gauss);
acc_gauss = trace(Conf_mat_gauss)/sum(sum(Conf_mat_gauss));
prec_mat_gauss = precision(Conf_mat_gauss);
recall_mat_gauss = recall(Conf_mat_gauss);





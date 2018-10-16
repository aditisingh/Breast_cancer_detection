classNames = {'0','1','2','3','4','5','6','7'};
rf = TreeBagger(120, TrainData_400, TrainData_targets_400, 'ClassNames',classNames);
label_rf = predict(rf,TestData_400);
label_rf = str2double(label_rf);
[c_rf,order_rf]=confusionmat(TestData_targets_400,label_rf);
sum=0;
for i=1:1:8
    for j=1:1:8
        sum = sum+c_rf(i,j);
    end
end
for i=1:1:8
    rows_sum = sumabs(Co(i,:));
    cols_sum = sumabs(Co(:,i));
    diagonal = Co(i,i);
    precision(i,1)= diagonal/cols_sum;
    recall(i,1)= diagonal/rows_sum;
end
acc = trace(Co)/sumabs(Co);
Co_2 = [sumabs(Co(1:4,1:4)) sumabs(Co(1:4,5:8));sumabs(Co(5:8,1:4)) sumabs(Co(5:8,5:8))];
for i=1:1:2
    rows_sum = sumabs(Co_2(i,:));
    cols_sum = sumabs(Co_2(:,i));
    diagonal = Co_2(i,i);
    precision(i,1)= diagonal/cols_sum;
    recall(i,1)= diagonal/rows_sum;
end
acc = trace(Co_2)/sumabs(Co_2);

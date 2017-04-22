data=[balanced_TrainData;balanced_TestData];
labels=[balanced_TrainData_targets;balanced_TestData_targets];
labels_1hot=zeros(size(labels,1),8);

idx_0=find(labels==0);
idx_1=find(labels==1);
idx_2=find(labels==2);
idx_3=find(labels==3);
idx_4=find(labels==4);
idx_5=find(labels==5);
idx_6=find(labels==6);
idx_7=find(labels==7);

for i=1:size(idx_0,1)
    labels_1hot(idx_0(i),:)=[0,0,0,0,0,0,0,1];
end

for i=1:size(idx_1,1)
    labels_1hot(idx_1(i),:)=[0,0,0,0,0,0,1,0];
end

for i=1:size(idx_2,1)
    labels_1hot(idx_2(i),:)=[0,0,0,0,0,1,0,0];
end

for i=1:size(idx_3,1)
    labels_1hot(idx_3(i),:)=[0,0,0,0,1,0,0,0];
end

for i=1:size(idx_4,1)
    labels_1hot(idx_4(i),:)=[0,0,0,1,0,0,0,0];
end

for i=1:size(idx_5,1)
    labels_1hot(idx_5(i),:)=[0,0,1,0,0,0,0,0];
end

for i=1:size(idx_6,1)
    labels_1hot(idx_6(i),:)=[0,1,0,0,0,0,0,0];
end

for i=1:size(idx_7,1)
    labels_1hot(idx_7(i),:)=[1,0,0,0,0,0,0,0];
end
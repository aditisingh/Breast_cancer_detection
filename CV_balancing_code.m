features = double(features);
targets = double(targets);
class0=[];
class1=[];
class2=[];
class3=[];
class4=[];
class5=[];
class6=[];
class7=[];
test_features = [features targets'];
for i=1:1:1820
    if targets(1,i)== 1
        class1 = [class1 ; test_features(i,:)];
        
    elseif targets(1,i)==2
        class2=[class2; test_features(i,:)];
        
    elseif targets(1,i)==3
        class3 = [class3; test_features(i,:)];
    
    elseif targets(1,i)==4
        class4=[class4; test_features(i,:)];
        
    elseif targets(1,i)==5
        class5 = [class5; test_features(i,:)];
        
    elseif targets(1,i)==6
        class6=[class6; test_features(i,:)];
        
    elseif targets(1,i)==7
        class7 = [class7; test_features(i,:)];
    
    elseif targets(1,i)==0
        class0 = [class0; test_features(i,:)];
        
    end
end
r1 = randperm(237,200);
r4 = randperm(788,225);
class1_RUS = [];
class4_RUS = [];
for i=1:1:200
    class1_RUS = [class1_RUS; class1(r1(1,i),:)];
end
for i=1:1:225
    class4_RUS = [class4_RUS; class4(r4(1,i),:)];
end

TrainData_400 = [class0(1:85,:); class1_RUS(1:160,:); class2(1:92,:); class3(1:104,:); class4_RUS(1:180,:); class5(1:109,:); class6(1:135,:); class7(1:110,:)];
TestData_400 = [class0(86:end,:); class1_RUS(161:end,:); class2(93:end,:); class3(105:end,:); class4_RUS(181:end,:); class5(110:end,:); class6(136:end,:); class7(111:end,:)];

TrainData_targets_400 = TrainData_400(:,4097);
TestData_targets_400 = TestData_400(:,4097);
TrainData_400 = TrainData_400(:,1:4096);
TestData_400 = TestData_400(:,1:4096);
function isttsvm_classifier(workpath,data)
%Realization of IST-TSVM

%input:
% workpath: the root directory where you put the code;
% data: the dataset,for example: data='BCI3' or data='BCI4'
% for example: isttsvm_classifier('C:\Program Files\MATLAB','BCI3')

dataSet = data;

classifier_algo = 'IST-TSVM';
  
confid_method = 'svm_value_center'; %the setting of confidential method
semi_method = 'self'; %self-training
selectedpercent = 0.5; %select 0.5 percent 

M_list = [10,15,20,25,30,35,40,45,50];%M=10,15,20,25,30,35,40,45,50
pos_neg_ratio_list = [1,4;2,3;1,1;3,2;4,1];%R=1:4,2:3,1:1,3:2,4:1

%R selected, if r_list =[3], M_list = [10,20,30,40,50];
%else if r_list = [1,2,4,5], M_list =  [10,15,20,25,30,35,40,45,50]
r_list = [1,2,4,5];

if strcmp(dataSet,'BCI3')
    disp('reading BCI competition III, data set IVa');
    load([workpath,'\tsvm\EEGdata\BCI_III_DSIVa\totalEEGSignals_BCI3.mat']);
elseif strcmp(dataSet,'BCI4')
    disp('reading BCI competition IV, data set IIa');
    load([workpath,'\tsvm\EEGdata\BCI_IV_DSIIa\totalEEGSignals_BCI4.mat']); 
end;

for r =r_list
    for M =M_list
        for iter = 1:10
            display('reading EEG data...');
            for s=1:length(totalEEGSignals)
                display(['No.' num2str(iter) ' iteration ' 'No.' num2str(s) 'subject' ' randomly distribute data']);
                M_pos = M*(pos_neg_ratio_list(r,1))/(pos_neg_ratio_list(r,1)+pos_neg_ratio_list(r,2));
                M_neg = M-M_pos;

                total_index = 1:length(totalEEGSignals{s}.y);
                temp_pos = totalEEGSignals{s}.left_index(iter,:);   
                temp_neg = totalEEGSignals{s}.right_index(iter,:);

                train_index = [temp_pos(1:M_pos),temp_neg(1:M_neg)];
                test_index = setdiff(total_index,train_index);
 
                tic;
                [best_testAcc,best_trainAcc,test_acc,train_acc] = semisupervised_iter(totalEEGSignals{s},train_index,test_index,classifier_algo,semi_method,confid_method,selectedpercent);
                result(s).testAcc(iter)= best_testAcc;
                subject{s}.test_acc_iter{iter} = test_acc;
                subject{s}.train_acc_iter{iter} = train_acc;
      
                trainingTime(s) = toc;
                result(s).trainingTime(iter) = trainingTime(s);
            end;
        end;

        meantestAcc = [];
        meantrainingTime = [];
        for s = 1:length(totalEEGSignals)
            result(s).meantestAcc =  mean(result(s).testAcc);
            result(s).stdtestAcc = std(result(s).testAcc);
            meantestAcc = [meantestAcc,result(s).meantestAcc];

            result(s).meantrainingTime = mean(result(s).trainingTime);
            result(s).stdtrainingTime = std(result(s).trainingTime);
            meantrainingTime = [meantrainingTime,result(s).meantrainingTime];
        end;

        stat.meantestAcc =  mean(meantestAcc)*100;
        stat.stdtestAcc = std(meantestAcc)*100;
        stat.meantrainingTime = mean(meantrainingTime);
        stat.stdtrainingTime = std(meantrainingTime);
        pos_neg = [num2str(pos_neg_ratio_list(r,1)) num2str(pos_neg_ratio_list(r,2))];
        fold = [workpath,'\tsvm\GeneratedData\'  dataSet  '_ratio_' pos_neg];
        if ~exist(fold)
            mkdir(fold);
        end;
        save([fold '\accuracy' '_M_' num2str(M) '_ratio_' pos_neg '_' dataSet '_' classifier_algo  'ги', num2str(stat.meantestAcc) ')' '.mat'],'result','stat','subject');
    end;
end;


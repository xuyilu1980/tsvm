function tsvm_classifier(workpath,data)
%compare different classifiers 

%input:
% workpath: the root directory where you put the code;
% data: the dataset,for example: data='BCI3' or data='BCI4'
% for example: tsvm_classifier('C:\Program Files\MATLAB','BCI3')

dataSet = data;

classifier_set = cell(1,7);
classifier_set{1} = 'SVM';
classifier_set{2} = 'TSVM_light';
classifier_set{3} = 'RTSVM';
classifier_set{4} = 'LDS';
classifier_set{5} = 'CCCP';
classifier_set{6} = 'CCCP1';  %CCCP algorithm with a new constraint
classifier_set{7} = 'ITSVM';

classifier_list = [1,3,4,5,7,2];  %Note that TSVM_light runs very long

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
        for i =classifier_list
            classifier_algo  = classifier_set{i};
            for iter = 1:10
                display('reading EEG data...');
                for s=1:length(totalEEGSignals)
                    display(['No.' num2str(iter) ' iteration ' 'No.' num2str(s) ' subject' ' randomly distribute data']);
                    M_pos = M*(pos_neg_ratio_list(r,1))/(pos_neg_ratio_list(r,1)+pos_neg_ratio_list(r,2));
                    M_neg = M-M_pos;

                    total_index = 1:length(totalEEGSignals{s}.y);
                    temp_pos = totalEEGSignals{s}.left_index(iter,:);
                    temp_neg = totalEEGSignals{s}.right_index(iter,:); 
                    train_index = [temp_pos(1:M_pos),temp_neg(1:M_neg)];
                    test_index = setdiff(total_index,train_index);
                    [Xl,Yl,Xu,Yu] = extractCSPFeatures_semi_one(totalEEGSignals{s},train_index,test_index);
 
                    Xl = Xl';
                    Xu = Xu';
                    XX=[Xl Xu];
                    yy=[Yl' -2*ones(1,length(Yu))];
                    ytest=Yu';
  
                    if strcmp(classifier_algo,'CCCP') || strcmp(classifier_algo,'CCCP1') || strcmp(classifier_algo,'RTSVM')
                        tic;
                        model=svmtrain1(Yl,Xl','-s 0 -t 0 -c 2');
                        if model.Label(1)==1,
                            W0=model.SVs'*model.sv_coef;
                            b0=-model.rho;
                        else
                            W0=-model.SVs'*model.sv_coef;
                            b0=model.rho;
                        end
                    end
  
                    
                    if strcmp(classifier_algo,'SVM')
                        tic;
                        yy5 = Yl;
                        xx5 = Xl';
                        xu5 = Xu';
                        ytest5 = ytest';
                        example_file = strcat(workpath,'\tsvm\temp\','train.data');
                        test_file = strcat(workpath,'\tsvm\temp\','test.data');
                        model_file = strcat(workpath,'\tsvm\temp\','pre.model');
                        output_file = strcat(workpath,'\tsvm\temp\','pre.mat');

                        save(example_file,'xx5','yy5');
                        svmlwrite(example_file,xx5,yy5);
                        options=svmlopt();
                        svm_learn(options,example_file,model_file);
                        save(test_file,'xu5','ytest5');
                        svmlwrite(test_file,xu5,ytest5);
                        svm_classify(options,test_file,model_file,output_file);
                        load(output_file,'-ascii');
                        predY=sign(pre);
                        SVM(s) =length(find(predY==ytest'))/length(ytest)
                        result(s).testAcc(iter)= SVM(s);
                    elseif strcmp(classifier_algo,'TSVM_light')
                        tic;
                        yy4 = [Yl' zeros(1,length(Yu))]';
                        pred=tsvm_l(XX',yy4,[workpath,'\tsvm\temp\'],'pre');
                        predY = sign(pred);
                        y4 = predY(length(Yl)+1:end);
                        TSVM_light(s) = length(find(y4==ytest'))/length(ytest)
                        result(s).testAcc(iter) = TSVM_light(s);
                    elseif strcmp(classifier_algo,'RTSVM')  
                        [w3 b3] = train_linear_transductive_svm_sg_robust_ls(XX,yy,4,4,W0,b0,0.1);
                        scores3=((Xu'*w3) + b3);
                        y3=sign(scores3);
                        RTSVM(s)=length(find(y3==ytest'))/length(ytest)
                        result(s).testAcc(iter)= RTSVM(s);
                    elseif strcmp(classifier_algo,'LDS')
                        tic;
                        opt.C = 0.1; % For this dataset, the classes overlap and it's
                        rho = 1;     % Anyvalue smaller than 3 is good 
                        Yp = lds(Xl,Xu,Yl,rho,opt);
                        te(s) = mean( Yp.*Yu < 0 );
                        LDS(s) = (100-(te(s)*100))/100;
                        result(s).testAcc(iter) = LDS(s);
                    elseif strcmp(classifier_algo,'CCCP')   
                        [w b] = train_linear_transductive_svm(XX,yy,2,2,W0,b0);
                        scores2=((Xu'*w) + b);
                        y2=sign(scores2);
                        CCCP(s) =length(find(y2==ytest'))/length(ytest)
                        result(s).testAcc(iter) = CCCP(s);
                   elseif strcmp(classifier_algo,'CCCP1')   
                        [w b] = train_linear_transductive_svm1(XX,yy,2,2,W0,b0);
                        scores2=((Xu'*w) + b);
                        y2=sign(scores2);
                        CCCP1(s) =length(find(y2==ytest'))/length(ytest)
                        result(s).testAcc(iter) = CCCP1(s);
                   elseif strcmp(classifier_algo,'ITSVM')
                        tic
                        opt.C = 0.1; % For this dataset, the classes overlap and it's
                        rho = 1;     % Anyvalue smaller than 3 is good 
                        XX_new = tsvm_cccp_lds_cos(Xl,Xu,Yl,rho,opt);
                        Xl_new = XX_new(:,1:size(Xl,2));
                        Xu_new = XX_new(:,size(Xl,2)+1:end);
                        model=svmtrain1(Yl,Xl_new','-s 0 -t 0 -c 2');
                        if model.Label(1)==1,
                          W1=model.SVs'*model.sv_coef;
                          b1=-model.rho;
                        else
                          W1=-model.SVs'*model.sv_coef;
                          b1=model.rho;
                        end
                        [w b] = train_linear_transductive_svm1(XX_new,yy,2,2,W1,b1);
                        scores2=((Xu_new'*w) + b);
                        y2=sign(scores2);
                        ITSVM(s) =length(find(y2==ytest'))/length(ytest)
                        result(s).testAcc(iter) = ITSVM(s);
                    end;
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
            fold = [workpath,'\tsvm\GeneratedData\' dataSet  '_ratio_' pos_neg];
            if ~exist(fold)
                mkdir(fold);
            end;
            save([fold '\accuracy' '_M_' num2str(M) '_ratio_' pos_neg '_' dataSet '_' classifier_algo  'ги', num2str(stat.meantestAcc) ')' '.mat'],'result','stat');
        end;
    end;
end;

function [best_testAcc,best_trainAcc,test_acc,train_acc] = semisupervised_iter(EEGSignals,train_index,test_index,classifier_algo,semi_method,confid_method,selectedpercent)
    nbFilterPairs =3;
    expanded_train_index = train_index;
    shunken_test_index = test_index;
    unchanged_total_index = union(train_index,test_index);
    best_testAcc = 0;
    best_trainAcc = 0;
    totalTrueY = EEGSignals.y;
    
 for iter = 1:5 
    CSPMatrix = learnCSPLagrangian_semi(EEGSignals,expanded_train_index);
    totalFeatures = extractCSPFeatures_semi(EEGSignals, CSPMatrix, nbFilterPairs,unchanged_total_index);
    trainFeatures = totalFeatures(expanded_train_index,:);
    testFeatures = totalFeatures(shunken_test_index,:);
    
    if strcmp(classifier_algo,'IST-TSVM')
        Xl = trainFeatures(:,1:2*nbFilterPairs);
        Yl = trainFeatures(:,end);
        Xu = testFeatures(:,1:2*nbFilterPairs);
        Yu = testFeatures(:,end);
        
        Xl = Xl';
        Xu = Xu';
        XX=[Xl Xu];
        yy=[Yl' -2*ones(1,length(Yu))];
        ytest=Yu';
        
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
        XX_origin = zeros(size(Xl_new,1),length(totalTrueY));
        XX_origin(:,expanded_train_index) = Xl_new;
        XX_origin(:,shunken_test_index) = Xu_new;
        pred=((XX_origin'*w) + b);
        y2=sign(pred)';
        train_accuracy = mean(totalTrueY(train_index) == y2(train_index));
        test_accuracy = mean(totalTrueY(test_index) == y2(test_index));       
    end
    
    train_acc(iter) = train_accuracy;
    test_acc(iter) = test_accuracy;
        
    if iter == 1 || abs(test_acc(iter)-test_acc(iter-1))>1/length(totalTrueY)
        if iter > 1 && abs(train_accuracy - best_trainAcc)>0.1
            break;
        else
            best_trainAcc = train_accuracy;
            best_testAcc = test_accuracy;
        end;
        
        if isempty(shunken_test_index)
            break;
        end;
        
        if strcmp(semi_method ,'self');
            expanded_train_index = train_index;
            shunken_test_index = test_index;
        end;
        
        if strcmp(confid_method,'svm_value_center')
            [addtrain_left,addtrain_right] = select_center_svm_value_center(shunken_test_index,expanded_train_index,pred,selectedpercent);
            if length(addtrain_left)>length(addtrain_right)
                min_add = length(addtrain_right);
            else min_add = length(addtrain_left);
            end;
            addtrain_left = addtrain_left(1:min_add);
            addtrain_right = addtrain_right(1:min_add);
        end;

        if strcmp(semi_method,'self');
            EEGSignals.y = totalTrueY;
            EEGSignals.y(addtrain_left) = 1;
            EEGSignals.y(addtrain_right) = -1;
            expanded_train_index = union(union(train_index,addtrain_left),addtrain_right);
            shunken_test_index = setdiff(setdiff(test_index,addtrain_left),addtrain_right); 
        end
    else break;
    end;
    

 end;
 
 if iter == 5
     best_testAcc = test_acc(iter);
     best_trainAcc = train_acc(iter);
 end;
      

  
 disp(['test set accuracy = ' num2str(best_testAcc) ' %']); 
 disp(['train set accuracy = ' num2str(best_trainAcc) ' %']);  

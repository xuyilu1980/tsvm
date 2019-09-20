function [ypred, accuracy] = rtsvm_predict_one_against_rest(test_labels, test_features, svm_models)

nbclass = length(svm_models);
n=size(test_features,2);

for i=1:nbclass,
    svm_model=svm_models{i};
    [score1]=test_features'*svm_model.w+svm_model.b;
    scores_all(i,:)=score1;
end

for i=1:n,
    [val ypred(i)]=max(scores_all(:,i));
end

dif=ypred-test_labels;
ll=find(abs(dif)>=0.5);
misclassifications=length(ll);
accuracy=(length(test_labels)-misclassifications)/length(test_labels);

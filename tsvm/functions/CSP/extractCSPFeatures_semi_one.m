function [Xl,Yl,Xu,Yu] = extractCSPFeatures_semi_one(EEGSignals,train_index,test_index)
nbFilterPairs = 3; %we use 3 pairs of filterse
CSPMatrix = learnCSPLagrangian_semi(EEGSignals,train_index);
%extracting CSP features from the training set
trainFeatures = extractCSPFeatures_semi(EEGSignals, CSPMatrix, nbFilterPairs,train_index);
%extracting CSP features from the testing set
testFeatures = extractCSPFeatures_semi(EEGSignals, CSPMatrix, nbFilterPairs,test_index);
Xl = trainFeatures(:,1:2*nbFilterPairs);
Yl = trainFeatures(:,end);
Xu = testFeatures(:,1:2*nbFilterPairs);
Yu = testFeatures(:,end);
end
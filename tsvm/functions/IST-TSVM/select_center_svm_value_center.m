function [addtrain_left,addtrain_right] = select_center_svm_value_center(shunken_test_index,expanded_train_index,pred,selectedpercent)
un_left_index = shunken_test_index(pred(shunken_test_index)>0);
un_right_index = shunken_test_index(pred(shunken_test_index)<0);
la_left_index = expanded_train_index(pred(expanded_train_index)>0);
la_right_index = expanded_train_index(pred(expanded_train_index)<0);
mean1 = mean(pred(la_left_index));
mean2 = mean(pred(la_right_index));

distance_mean1 = abs(pred(un_left_index)-mean1)./abs(pred(un_left_index));
[~,l1] = sort(distance_mean1,'ascend');
addtrain= un_left_index(l1);
addtrain_left = addtrain(1:floor(length(un_left_index)*selectedpercent));

distance_mean2 = abs(pred(un_right_index)-mean2)./abs(pred(un_right_index));
[~,r1] = sort(distance_mean2,'ascend');
addtrain = un_right_index(r1);
addtrain_right = addtrain(1:floor(length(un_right_index)*selectedpercent));

%The classification performance of BCI3 and BCI4, with varying the size
% of the balanced labelled sets

x=[10,20,30,40,50];

%BCI3,pos_neg_ratio = 1
% y=[66.36	76.42	79.87	81.36	84.15
% 67.09	76.56	81.22	83.32	85.28
% 64.12	73.25	76.28	79.13	82.64
% 63.27	73.65	77.86	80.83	84.14
% 67.3	76.44	79.78	80.33	82.21
% 67.47	76.48	80.01	81.04	82.43
% 68.07	76.89	81.24	81.97	83.5];

%BCI4, pos_neg_ratio = 1
y =[67.91	70.64	72.13	73.25	74.08
68.41	69.56	71.11	71.7	72.9
68.47	70.18	71.04	71.12	72.01
68.15	70.05	71.01	72.21	73.29
68.78	72.48	73.16	73.58	74.27
69.1	72.66	72.59	73.51	74.21
70.22	72.23	73.17	73.67	73.89];

y_SVM = y(1,:);
y_TSVM = y(2,:);
y_RTSVM = y(3,:);
y_LDS = y(4,:);
y_CCCP = y(5,:);
y_ITSVM = y(6,:);
y_IST_TSVM = y(7,:);
%   set(gcf,'position',[100 100 1000 600]);
plot(x,y_SVM,'->g',x,y_TSVM,'-sr',x,y_RTSVM,'-+c',x,y_LDS,'-^y',x,y_CCCP,'-om',x,y_ITSVM,'-*b',x,y_IST_TSVM,'-dk');
legend('SVM','TSVM-light','RTSVM','LDS','CCCP','ITSVM','IST-TSVM');
xlabel('Number of labelled trials');
ylabel('Classification Accuracy (%)');

set(gca,'FontSize',16);

 set(0,'defaultfigurecolor','w');

% tH = title('Data set II-a R = 1:1');
% tH = title('Data set IV-a R = 1:1');



x=[10,20,30,40,50];

%BCI3,pos_neg_ratio = 1

%BCI3纯新的
% y=[66.36	76.42	79.87	81.36	84.15
% 67.08	76.55	81.22	83.32	85.33
% 64.15	73.2	76.27	79.16	82.64
% 63.27	73.65	77.86	80.83	84.14
% 67.3	76.44	79.78	80.33	82.21
% 67.47	76.48	80.01	81.04	82.43
% 68.07	76.89	81.24	81.97	83.5];

%BCI4, pos_neg_ratio = 1
%新的
y =[67.91	70.64	72.13	73.25	74.08
68.4	69.56	71.12	71.7	72.9
68.49	70.19	71.03	71.13	72.01
68.15	70.05	71.01	72.21	73.29
68.78	72.48	73.16	73.58	74.27
69.1	72.66	72.59	73.51	74.21
70.22	72.23	73.17	73.67	73.89];




% C=[0.904705882352941,0.191764705882353,0.198823529411765;0.294117647058824,0.544705882352941,0.749411764705882;0.371764705882353,0.717647058823529,0.361176470588235;1,0.548235294117647,0.100000000000000;0.865000000000000,0.811000000000000,0.433000000000000;0.685882352941177,0.403529411764706,0.241176470588235;0.971764705882353,0.555294117647059,0.774117647058824];
y_SVM = y(1,:);
y_TSVM = y(2,:);
y_RTSVM = y(3,:);
y_LDS = y(4,:);
y_CCCP = y(5,:);
y_ITSVM = y(6,:);
y_IST_TSVM = y(7,:);
%   set(gcf,'position',[100 100 1000 600]);
plot(x,y_SVM,'->g',x,y_TSVM,'-sr',x,y_RTSVM,'-+c',x,y_LDS,'-^y',x,y_CCCP,'-om',x,y_ITSVM,'-*b',x,y_IST_TSVM,'-dk');
% plot(x,y_SVM,'->',x,y_TSVM,'-.sr',x,y_RTSVM,'-+',x,y_LDS,'-.^',x,y_CCCP,'-om',x,y_LDS_CCCP,'-*b',x,y_STLDS_CCCP,'-dk');
% plot(x,y_SVM,'-.>',x,y_TSVM,'-s',x,y_RTSVM,'-+',x,y_LDS,'--^',x,y_CCCP,'-o',x,y_LDS_CCCP,'-*',x,y_STLDS_CCCP,'-d');

% plot(x,y_csp,'->k',x,y_COV_MDM,'-*k',x,y_ccsp1,'-+k',x,y_COV_MDM_TL,'-sk',x,y_wtrcsp,'-ok',x,y_COV_MDM_SSTL,'-^k');%全用黑色线条
 legend('SVM','TSVM-light','RTSVM','LDS','CCCP','ITSVM','IST-TSVM');
 xlabel('Number of labelled trials');
 ylabel('Classification Accuracy (%)');

 set(gca,'FontSize',18);

 set(0,'defaultfigurecolor','w');

% tH = title('Data set II-a R = 1:1');
% tH = title('Data set IV-a R = 1:1');

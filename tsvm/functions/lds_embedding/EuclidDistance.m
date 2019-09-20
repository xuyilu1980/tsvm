function R=EuclidDistance(X,mu);

% This function computes ||x_n-mu_j||^2.
% Inputs
%   X   -   d x N dimensional matrix whose columns are observation samples x_n
%   mu  -   d x Nh dimensional matrix
%   sig -   Nh x 1 dimensional row vector
% Outputs
%   R   -   N x Nh dimensional matrix that includes computed values
%   
%   Hakan CEVIKALP INRIA-RHONE ALPES 9/13/2006

[d,Nh]=size(mu);%�õ�ά��
M=size(X,2); %�õ�������
X2=X.^2; %��X��ƽ��ֵ
SumTerm=sum((mu.^2));
R = -2*X'*mu + X2'*ones(d,Nh) + ones(M,1)*SumTerm;%�õ�����ŷʽ�����ƽ��


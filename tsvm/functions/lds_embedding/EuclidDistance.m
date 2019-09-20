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

[d,Nh]=size(mu);%得到维度
M=size(X,2); %得到样本数
X2=X.^2; %求X的平方值
SumTerm=sum((mu.^2));
R = -2*X'*mu + X2'*ones(d,Nh) + ones(M,1)*SumTerm;%得到的是欧式距离的平方


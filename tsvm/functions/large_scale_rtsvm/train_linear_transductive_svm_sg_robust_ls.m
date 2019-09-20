
function [w b costHist] = train_linear_transductive_svm_sg_robust_ls(X,y,C1,C2,w0,b0,alfa,cf,tol, nIter, ss)


s = -0.20;
if nargin < 7, alpha = 0.001; end
if nargin < 8, cf = 1; end 
if nargin < 9, tol = 0.001; end
if nargin < 10, nIter = 5; end
if nargin >= 11, s = ss; end

% finding unlabled data
unlabeled=find(y==-2);
pos=find(y==1);
neg=find(y==-1);
npos=length(pos);
nneg=length(neg);
L=length(pos)+length(neg);
U=length(unlabeled);
XX=[X(:,pos) X(:,neg) X(:,unlabeled) X(:,unlabeled)];  
yy=[ones(1,length(pos)) -ones(1,length(neg)) ones(1,length(unlabeled)) -ones(1,length(unlabeled))];  


nn=size(XX,2);
beta=zeros(1,L+2*U); 

scores = ((XX'*w0) + b0).*yy';
ll=find(scores<s);
beta(ll)=C1;   
kk=find(ll>L);
beta(ll(kk))=C2;

costHist = [];
w=w0;
b=b0;
for it=1:nIter
    it
    wp=w;
    bp=b;
    [w b cost]=transductive_linear_svm_sg(XX,yy,L,U,w,b,C1,C2,beta,alfa,cf);
%    cost
    costHist = [costHist cost];
    if norm(w-wp)<tol,
        break;
    end
    % Finding new beta values
    beta=zeros(1,L+2*U);
    scores = ((XX'*w) + b ).*yy';
    ll=find(scores<s); 
    beta(ll)=C1;
    kk=find(ll>L);
    beta(ll(kk))=C2;
end


function [w b costHist]=transductive_linear_svm_sg(XX,yy,L,U,w,b,C1,C2,beta,alfat,cf)

n=length(yy);
%% balancing constr: W'*mu+W0=gamma£¬
mean_unlabeled=full( mean(XX(:,L+1:end)')');
mu=[mean_unlabeled];
gamma=sum(yy(1:L))/L;
norm_mu=norm(mu)^2+1; 
d=size(XX,1);
n=L+(2*U);
cost=Inf;
tol=10^-4;
nUpdates = 0;
T = 700;
costHist = [];
for t=1:T,
    alfa=alfat/t;
    idx=randperm(L+2*U);
    if ~issparse( XX )
        [w b]=rtsvm_sgd_step_new(XX,yy,idx,w,b,alfa,[C1 C2],mu,gamma,beta,L,U, nUpdates,cf);
    else
        [w b]=rtsvm_sgd_step_new_sparse(XX,yy,idx,w,b,alfa,[C1 C2],mu,gamma,beta,L,U, nUpdates,cf); 
    end
    scores=((XX'*w) + b).*yy';
    ll=find(scores<1); 
    kk=find(ll<=L);
    c1=sum(1-scores(ll(kk)))*C1/n;  
    c2=(beta(1:L)*(scores(1:L)')')/n;
    kk=find(ll>L);
    c3=sum(1-scores(ll(kk)))*C2/n;
    c4=(beta(L+1:end)*(scores(L+1:end)')')/n;
    costn=0.5*norm(w)^2+(c1+c2+c3+c4); 
    costHist = [costHist costn];
    fprintf('\n cost is = %.4f',costn); 
    if (cost-costn)<tol; 
        break;
    end
    cost=costn;
end
        

        
            
    





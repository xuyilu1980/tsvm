function [w b new_alpha] = train_linear_transductive_svm(X,y,C1,C2,w0,b0)
s = -0.20;  
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
y0=1;
x0=sum(X(:,unlabeled)')'/U; 
XX=[x0 XX];
yy=[y0 yy];
f=yy;
f(1)=sum(yy(2:L+1))/L;  %  traditional constraint
nn=size(XX,2);
beta=zeros(1,2*U);  
scores = ((XX'*w0) + b0).*yy'; 
ll=find(scores(L+2:nn)<s);
beta(ll)=C2;
kernel.type=0;

b=b0;

for it=1:5 
    it
    Aeq=[ones(1,L+(2*U)+1)];  
    beq=0;
    LB=[-1 zeros(1,npos)    -C1*ones(1,nneg)   -beta(1,1:U)   beta(1,U+1:2*U)-C2];
    UB=[1   C1*ones(1,npos)   zeros(1,nneg)     C2-beta(1,1:U) beta(1,U+1:2*U)]; %upper bound
   
    yyy=ones(1,length(yy));
    [new_alpha,faval]=mlcv_quadprog(XX',yyy,kernel,-f,Aeq,beq,LB,UB);
    
   
    faval
    
    w=XX*new_alpha;
    
    % Finding bias
    if 1
        flag1=0;
        flag2=0;
        ll=find(abs(new_alpha(2:npos+1))>10^-5); 
        
        if isempty(ll)~=1,
            scores=w'*XX(:,ll+1);
            b1=1-mean(scores.*yy(ll+1))
            flag1=1;
        end
        ll2=find(abs(new_alpha(npos+2:npos+nneg+1))>10^-5);
        if isempty(ll2)~=1
            scores=w'*XX(:,ll2+1+npos);
            b2=-1-mean(scores);
            flag2=1;
        end
        if flag1==1 & flag2==1
            b=mean([b1 b2])
        elseif flag1==1
            b=b1;
        elseif flag2==1
            b=b2;
        else
            scores=w'*XX(:,2:end);
            b=-mean(scores);
        end
        
    end
    
    
    % Finding new beta values
    beta=zeros(1,2*U);
    scores = ((XX'*w) + b ).*yy';
    ll=find(scores(L+2:nn)<s); 
    beta(ll)=C2;

end

close all







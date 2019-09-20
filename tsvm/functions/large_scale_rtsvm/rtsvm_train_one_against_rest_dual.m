function rtsvm_models = rtsvm_train_one_against_rest_dual(X,y,nbclass,Cvec)

for i=1:nbclass,
    ll=find(y==-2);
    ld=find(y~=-2);
    yone=(y==i)+(y~=i)*-1;
    yone(ll)=-2;
    
    cc=Cvec(1);
    sc=num2str(cc);
    model=svmtrain(yone(ld),X(:,ld)',['-s 0 -t 0 -c ' sc ]);
    W0=model.SVs'*model.sv_coef;
    b0=-model.rho;
    [W b] = train_linear_transductive_svm_robust_dual(X,yone,Cvec(1),Cvec(2),W0,b0);
    hyperplane.w=W;
    hyperplane.b=b;
    rtsvm_models{i} = hyperplane;

end


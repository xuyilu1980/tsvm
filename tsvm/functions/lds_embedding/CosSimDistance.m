function R=CosSimDistance(X)
[d,M]=size(X);
R = zeros(M);
for i = 1:M
    for j = 1:i
        t = dot(X(:,i),X(:,j))/(norm(X(:,i))*norm(X(:,j)));
        R(i,j) = 1-t;
        R(j,i) = R(i,j);
    end;
    R(i,i) = 0;
end;
% (C) A. Zien and O. Chapelle, MPI for biol. Cybernetics, Germany

function [ NN, D2 ] = calcNnDists_lap( X, D2full, nofNn, annEps )  %D2得到一个对角线元素几乎都为0，且对称的矩阵
%nofNn计算几个邻居，nofNn=0就是获得全连接矩阵
%以下是拉普拉斯矩阵的求解
options=make_options('gamma_I',1,'gamma_A',1e-5,'NN',6,'KernelParam',0.35);
options.Verbose=1;
options.UseBias=1;
options.UseHinge=1;
options.LaplacianNormalize=0;
options.NewtonLineSearch=0;

assert( nargin == 4 );
if( nofNn == 0 | annEps < 0 )
  % --- obtain full distance matrix
  if( length(D2full) == 0 )
    if( issparse( X ) )
      D2full = EuclidDistance( full(X),  full(X));  %等于它在这里换成了matlab的实现方式，lds文件夹中是用了C语言的实现方式
    else
%       D2full =  EuclidDistance( X, X ); %得到一个对角线元素几乎都为0，且对称的矩阵
      D2full = adjacency(options,X');
    end;
  end;
  % --- find nearest neighbors by sorting, if requested
  if( nofNn == 0 )
    NN = int32( [] );
    D2 = D2full;
  else
    [ D2, NN ] = sort( D2full );%好像并没有执行到这一步,单独对每一列进行排序
    m = size( D2full, 1 );
    assert( all( D2(1,:) == 0 ) );
    % - are all points their own nearest neighbors?
    idx = find( NN(1,:) ~= 1:m );
    if( length( idx ) > 0 )
      for( i = idx )
	j = find( NN(:,i) == i );
	assert( length(j) == 1 );
	NN(j,i) = NN(1,i);
	NN(1,i) = i;
      end;
    end;
    % - now, all points must be their own nearest neighbors.
    assert( all( NN(1,:) == 1:m ) );
    selected = 1 + (1:nofNn);%从第2行开始，选择每一列所对应的标签样本中，最接近的nofNn个邻居
    D2 = D2( selected, : );
    NN = int32( NN( selected, : ) );%得到每一列所对应的标签样本中，最接近的nofNn个邻居的索引号
  end;
else
  % --- approximate nearest neighbors
  if( issparse( X ) )
    [ NN, D2 ] = annk( full(X), nofNn, annEps );
  else
    assert( ~ issparse( X ) );
    [ NN, D2 ] = annk( X, nofNn, annEps );
  end;
end;



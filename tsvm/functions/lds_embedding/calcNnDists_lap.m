% (C) A. Zien and O. Chapelle, MPI for biol. Cybernetics, Germany

function [ NN, D2 ] = calcNnDists_lap( X, D2full, nofNn, annEps )  %D2�õ�һ���Խ���Ԫ�ؼ�����Ϊ0���ҶԳƵľ���
%nofNn���㼸���ھӣ�nofNn=0���ǻ��ȫ���Ӿ���
%������������˹��������
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
      D2full = EuclidDistance( full(X),  full(X));  %�����������ﻻ����matlab��ʵ�ַ�ʽ��lds�ļ�����������C���Ե�ʵ�ַ�ʽ
    else
%       D2full =  EuclidDistance( X, X ); %�õ�һ���Խ���Ԫ�ؼ�����Ϊ0���ҶԳƵľ���
      D2full = adjacency(options,X');
    end;
  end;
  % --- find nearest neighbors by sorting, if requested
  if( nofNn == 0 )
    NN = int32( [] );
    D2 = D2full;
  else
    [ D2, NN ] = sort( D2full );%����û��ִ�е���һ��,������ÿһ�н�������
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
    selected = 1 + (1:nofNn);%�ӵ�2�п�ʼ��ѡ��ÿһ������Ӧ�ı�ǩ�����У���ӽ���nofNn���ھ�
    D2 = D2( selected, : );
    NN = int32( NN( selected, : ) );%�õ�ÿһ������Ӧ�ı�ǩ�����У���ӽ���nofNn���ھӵ�������
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



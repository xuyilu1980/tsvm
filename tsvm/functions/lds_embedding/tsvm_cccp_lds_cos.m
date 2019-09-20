function XX_new = tsvm_cccp_lds_cos(Xl,Xu,Yl,rho,opt)

%   Combining CCCP with LDS 
%
% Xl:  d x n matrix with the labeled points
% Xu:  d x m matrix with the unlabeled points
% Yl:  column vector of length n containing the labels (+1 or -1 for binary)
% rho: the constant in (2)
% opt: optional structure containing the (optional) fields,
%         C:       the soft margin parameter relative to 1/var^2
%                  [default = 1]
%         nofNn:   number of NN in the graph construction 
%                  [default = 0, i.e. fully connected graph]
%         sigma:   the width of the RBF kernel appearing at the end
%                  of 2.1.2, relative to the default value [default = Inf] 
%         delta:   the threshold in (6) [default = 0.1]
%         nofIter: number of iterations for C* to reach C [default = 10]
%         Cinit:   initial value for C* relative to C [default = 0.01] 
%         Cfinal:  final value of C* relative to C [default = 1]
%         opt_tb:  uses the optimization toolbox if available [default = 1]
%         maxiter: maximum number of iterations in each gradient
%                  descent (multiplied by nb of variables) [default = 3]
%         tolfun:  stopping criterion on the function value
%                  (relative to C) [default = 1e-5]
%         verb:    verbosity [default = 1]
%         splits:  estimate the test error by cross-validation
%                  splits is a cell array containing the indices of the points
%                  left out [default = leave-one-out].
% Yu: is a real valued output
% err: estimated error by cross-validation [cf opt.splits]
%  
% (C) O. Chapelle and A. Zien, MPI for biol. Cybernetics, Germany


% Fill up default values for opt
if ~exist('opt','var')
  opt = [];
end;
if ~isfield(opt,'delta')       opt.delta       = 0.1;       end;
if ~isfield(opt,'nofIter')     opt.nofIter     = 10;        end;  
if ~isfield(opt,'C')           opt.C           = 1;         end;
if ~isfield(opt,'Cinit')       opt.Cinit       = 0.01;      end;
if ~isfield(opt,'Cfinal')      opt.Cfinal      = 1;         end;
if ~isfield(opt,'nofNn')       opt.nofNn       = 0;         end;
if ~isfield(opt,'sigma')       opt.sigma       = Inf;       end;
if ~isfield(opt,'opt_tb')      opt.opt_tb      = 1;         end;
if ~isfield(opt,'maxiter')     opt.maxiter     = 3;         end;
if ~isfield(opt,'tolfun')      opt.tolfun      = 1e-5;      end;
if ~isfield(opt,'verb')        opt.verb        = 1;         end;

% Check the arguments
[ d0, m0 ] = size( Xl );
[ d1, m1 ] = size( Xu );
if (d0 ~= d1) | ((m0 ~= length(Yl)) & (m0 + m1 ~= length(Yl)))
  error('Dimensions mismatch');
end;

m = m0 + m1;
X = [ Xl , Xu ];
clear Xl;
clear Xu;

% assert( 0 <= opt.delta && opt.delta < 1 );
assert( 1 <= opt.nofIter );
assert( 0 < opt.C );
assert( 0 < opt.Cinit && opt.Cinit <= opt.Cfinal );
assert( 0 <= opt.nofNn && opt.nofNn <= m );
assert( 0 < opt.sigma );

if (size(Yl,2) ~= 1)
  error(['Third argument should be the label vector']);
end;
ks = unique( Yl );
k = length( ks );
if( k < 2 )
  error(['At least two different classes (labels in Yl) required']);
end;
if( k == 2 )
  if any(abs(Yl)~=1) 
    error(['Labels in binary problems must be +/-1']);
  end;
end;

if opt.verb >= 2  
  fprintf('%d classes, %d labeled points, %d unlabeled, in %d dimension(s)\n',k,m0,m1,d0);
end;

% Compute the full distance graph ...
param.annEps = -1;  % do not use ANN (approximate NN)
if opt.verb >= 1
  if opt.nofNn == 0 
    fprintf('Computing  distances ...\n');
  else
    fprintf('Computing %d nearest neighbors graph ...\n', opt.nofNn);
    end;
end;
[ NN, D2 ] = calcNnDists_cos( X, [], opt.nofNn, param.annEps );

% ... and the new metric
param.pathNorm = rho;
param.sigma = +inf;
param.nofNn = opt.nofNn;
if opt.verb >= 1
  fprintf('Computing shortest paths ...\n'); 
end;

pos_index = find(Yl == 1);
neg_index = find(Yl == -1);
D2(pos_index,neg_index)=inf;
D2(neg_index,pos_index)=inf;
D2( isinf( D2 ) ) = 2 * max( D2( ~isinf( D2 ) ) ); 

E2 = graphDistKernelC( X, D2, NN, 1:m, param ); 
if( opt.nofNn ~= 0 )
  E2 = min( E2,  E2' );
  E2( isinf( E2 ) ) = 2 * max( E2( ~isinf( E2 ) ) );
end;

% Compute the new kernel, do the MDS reduction, ...
defaultSigma = calcDefaultSigma( E2(:), 2 );
if opt.verb >= 2
  fprintf( '  default sigma = %.3f\n', defaultSigma );
end;
sigma = opt.sigma * defaultSigma;
if isinf(sigma)
  if any(isinf(E2(:)))
    error(['Cannot do the MDS: the graph is not connected'])
  end;

  [Xmds,E] = mds(E2,opt); 
else
  param.rbf = 'gauss';
  K = calcRbfKernel( E2, param.rbf, sigma, 1 );
  [Xmds,E] = mds(2-2*K,opt); 
end;                       


if opt.delta >= 0
  cond1 = 1-cumsum(E)/sum(E) < opt.delta;
  cond2 = E < opt.delta*E(1);
  nb_comp = min( find( cond1 & cond2 ) ) - 1;
else
  nb_comp = length(E);
end;
if opt.verb >= 2
  fprintf('  %d components kept\n',nb_comp);
end;

XX_new = Xmds(:,1:nb_comp); % Keep only the first components
XX_new = XX_new';
XX_new = [XX_new;X];  %comprehensive feature

function [Y, L] = mds(D,opt)
% Classical Multidimensional Scaling.

if opt.verb >= 1, fprintf('Computing MDS ...\n'); end;
n = length(D);

if opt.delta>=0
  H = eye(n) - repmat(1/n,n,n); 
  K = H * (-.5 * D) * H;
  [V L] = eig((K+K')./2);
  [L ind] = sort(-diag(L)); 
  L = -L;
  
  keep = find(L > 1e-10*max(L));
  Y = V(:,ind(keep)) * diag(sqrt(L(keep)));
else
  k = -opt.delta;
  H = eye(n) - repmat(1/n,n,n);
  K = H * (-.5 * D) * H;
  K = (K+K')/2;
  
  [V L] = eig(K(1:k,1:k));
  L = diag(L);
  keep = find(L > 1e-10*max(L)); 
  V = V(:,keep) .* repmat(L(keep)'.^(-1/2),k,1);
  Y = K(:,1:k)*V;
  L = L(keep);
end;
  
  


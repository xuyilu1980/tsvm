function [x,stat] = mlcv_quadprog(X, y, kernel, f, a, b, LB, UB, x0, opt)
% MLCV_QP_GSMO Generalized SMO algorithm.
%
% Synopsis:
%  [x,stat] = libqp_gsmo(X,y,kernel,f,a,b,LB,UB)
%  [x,stat] = libqp_gsmo(X,y,kernel,f,a,b,LB,UB,x0)
%  [x,stat] = libqp_gsmo(X,y,kernel,f,a,b,LB,UB,[],opt)
%  [x,stat] = libqp_gsmo(X,y,kernel,f,a,b,LB,UB,x0,opt)
% 
% Description:
%  This function implements the Generalized SMO algorithm which solves
%  the following convex quadratic programming task:
%
%   min 0.5*x'*H*x + f'*x  （其中H = X'*X）
%    x                                      
%
%   subject to   a'*x = b 
%                LB(i) <= x(i) <= UB(i)   for all i=1:n %x(i)应该是x对应的y值
%
% Reference:
%  S.-S. Keerthi, E.G.Gilbert. Convergence of a Generalized SMO Algorithm for SVM 
%   Classifier Design. Technical Report CD-00-01, Control Division, Dept. of Mechanical 
%   and Production Engineering, National University of Singapore, 2000. 
%   http://citeseer.ist.psu.edu/keerthi00convergence.html   
%
% Input:
%  X [n x d] Input matrix  n代表样本的个数，d代表维数
%  y [n x 1] Label vector
%
%  kernel [struct]
%       kernel.type = (default 0)       
%                   0 -- linear: u'*v\n"
%                   1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
%                   2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
%                   3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
%                   4 -- intersection: sum(min(x,y))\n"
%                   5 -- chi-square : sum( (xi - yi)^2 / (2*(xi + yi)))\n"
%                   6 -- rbf-chi-square : exp(-gamma*(sum( (xi - yi)^2 / (2*(xi + yi))))\n"
%       kernel.degree = (default 2)		
%		kernel.gamma  = (default 1)       
%		kernel.coef0  = (default 0)  
%
%  f [n x 1] Vector.
%  a [n x 1] Vector which must not contain zero entries.
%  b [1 x 1] Scalar.
%  LB [n x 1] Lower bound; -inf is allowed.
%  UB [n x 1] Upper bound; inf is allowed.
%
% Optional inputs: 
%  x0 [n x 1] Initial solution.
%
%  options [struct] 
%    .TolKKT [1 x 1] Determines relaxed KKT conditions (default TolKKT=0.001);
%       it correspondes to $\tau$ in Keerthi's paper.
%    .verb [1 x 1] if > 0 then prints info every verb-th iterations (default 0)
%    .MaxIter [1 x 1] Maximal number of iterations (default inf).
%  
% Output:
%  x [n x 1] Solution vector.
%  stat [struct]
%   .QP [1x1] Primal objective value.
%   .exitflag [1 x 1] Indicates which stopping condition was used:
%      nIter >= MaxIter                  ->  exitflag = 0
%      relaxed KKT conditions satisfied  ->  exitflag = 4  
%   .nIter [1x1] Number of iterations.
%
% Example:
%  n=50; X = rand(n,n); H = X'*X; f = -10*rand(n,1); 
%  a=rand(1,n)+0.1; b=rand; tmp = rand(n,1); LB = tmp-1; UB = tmp+1;
%
%  tic; [x1,stat1] = libqp_gsmo(H,f,a,b,LB,UB); fval1=stat1.QP, toc
%  tic; [x2,fval2] = quadprog(H,f,[],[],a,b,LB,UB); fval2, toc
% 
%
% Copyright (C) 2010 Huseyin Gunduz
% MLCV Laboratory

% options
if nargin < 8, error('At least six input arguments are required.'); end
if nargin < 9 || isempty(x0), 
  % create feasible solution x0 if not given
  x0 = zeros(length(f),1); 
  xa = 0; i = 0;
  while x0'*a(:) ~= b,
    i = i + 1;
    if i > length(a),
      error('No feasible solution exists.');
    end
    x0(i) = min(UB(i),max(LB(i),(b-xa)/a(i)));
    xa = xa + x0(i)*a(i);
  end
end

%keyboard
if nargin < 10, opt = []; end
if ~isfield(opt,'TolKKT'), opt.TolKKT = 0.001; end
if ~isfield(opt,'MaxIter'), opt.MaxIter = inf; end
if ~isfield(opt,'verb'), opt.verb = 0; end



%default kernel parameters
if ~isfield(kernel,'type'), kernel.type = 0; end
if ~isfield(kernel,'degree'), kernel.degree = 2; end
if ~isfield(kernel,'gamma'), kernel.gamma = 1; end
if ~isfield(kernel,'coef0'), kernel.coef0 = 0; end


% mex function entry point
[x,stat.QP,stat.exitflag,stat.nIter] = mlcv_qp_gsmo_mex(X,... 
                                                      y,... 
                                                      kernel.type,... 
                                                      kernel.degree,...
                                                      kernel.gamma,...
                                                      kernel.coef0,...
                                                      f,...
                                                      a,...
                                                      b,...
                                                      LB,...
                                                      UB,...
                                                      x0,...
                                                      opt.MaxIter,...
                                                      opt.TolKKT,...
                                                      opt.verb);

return;
% EOF

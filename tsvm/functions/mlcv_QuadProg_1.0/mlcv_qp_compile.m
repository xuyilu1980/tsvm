% This script compiles Matlab's MEX interfaces for LIBQP solvers.
%
clc
mex -largeArrayDims libqp_gsmo_mex.c libqp_gsmo.cpp
%mex mlcv_qp_gsmo_mex.c mlcv_qp_gsmo.cpp


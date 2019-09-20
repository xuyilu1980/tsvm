MEX := mex
MEXFLAGS := $(shell if uname -m | grep -q x86_64 ; then echo -largeArrayDims ; fi)
MEXSUFFIX := $(shell if uname -m | grep -q x86_64 ; then echo mexa64 ; else echo mexglx ; fi)

MEXFLAGS += -lm

all: rtsvm_sgd_step_new_sparse.$(MEXSUFFIX) rtsvm_sgd_step_new.$(MEXSUFFIX)

rtsvm_sgd_step_new_sparse.$(MEXSUFFIX):	rtsvm_sgd_step_new_sparse.cpp
		$(MEX) $(MEXFLAGS) -I../ $(INCLUDES) -DMATLAB -O -output $@ rtsvm_sgd_step_new_sparse.cpp

rtsvm_sgd_step_new.$(MEXSUFFIX):	rtsvm_sgd_step_new.cpp
		$(MEX) $(MEXFLAGS) -I../ $(INCLUDES) -DMATLAB -O -output $@ rtsvm_sgd_step_new.cpp

clean: 
		rm -f *~ *$(MEXSUFFIX)

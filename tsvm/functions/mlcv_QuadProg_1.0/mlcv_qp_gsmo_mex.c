/*--------------------------------------------------------------------------
 gsmo_mex.c: Matlab MEX interface for the Generalized SMO solver.

 Synopsis:
  [x,QP,exitflag,nIter] = ...
     libqp_gsmo_mex(X,y,kernel,f,a,b,LB,UB,x0,MaxIter,TolKKT,verb)

 Compile:
  mex gsmo_mex.c gsmolib.c

 Description:
   See "help libqp_gsmo"
                                                                    
 Copyright (C) 2006-2008 Vojtech Franc, xfrancv@cmp.felk.cvut.cz
 Center for Machine Perception, CTU FEL Prague

-------------------------------------------------------------------- */

#include <string.h>
#include <stdint.h>
#include "mex.h"
#include "mlcv_qp.h"

#define LIBQP_MATLAB

#define INDEX(ROW,COL,NUM_ROWS) ((COL*NUM_ROWS)+ROW)
#define MIN(A,B) ((A < B) ? A : B)
#define MAX(A,B) ((A > B) ? A : B)

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

/* ------------------------------------------------------------*/
/* Declaration of global variables                             */
/* ------------------------------------------------------------*/
double *mat_H;
uint32_t nVar;

struct svm_parameter param;		/* set by parse_command_line */
struct svm_problem prob;			/* set by read_problem */
struct svm_node *x_space;


/* read in a problem (in svmlight format) */
void read_problem_dense(const mxArray *label_vec, const mxArray *instance_mat)
{
	int i, j, k;
	int elements, max_index, sc, label_vector_row_num;
	double *samples, *labels;

	prob.x = NULL;
	prob.y = NULL;
	x_space = NULL;

	labels = mxGetPr(label_vec);
	samples = mxGetPr(instance_mat);
	sc = (int)mxGetN(instance_mat);

	elements = 0;
	/* the number of instance */
	prob.l = (int)mxGetM(instance_mat);
	label_vector_row_num = (int)mxGetM(label_vec);

	if(label_vector_row_num!=prob.l)
	{
		mexPrintf("Length of label vector does not match # of instances.\n");
		/*return -1;*/
	}

	if(param.kernel_type == PRECOMPUTED)
		elements = prob.l * (sc + 1);
	else
	{
		for(i = 0; i < prob.l; i++)
		{
			for(k = 0; k < sc; k++)
				if(samples[k * prob.l + i] != 0)
					elements++;
			/* count the '-1' element */
			elements++;
		}
	}

	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct svm_node *,prob.l);
	x_space = Malloc(struct svm_node, elements);

	max_index = sc;
	j = 0;
	for(i = 0; i < prob.l; i++)
	{
		prob.x[i] = &x_space[j];
		prob.y[i] = labels[i];

		for(k = 0; k < sc; k++)
		{
			if(param.kernel_type == PRECOMPUTED || samples[k * prob.l + i] != 0)
			{
				x_space[j].index = k + 1;
				x_space[j].value = samples[k * prob.l + i];
				j++;
			}
		}
		x_space[j++].index = -1;
	}

	if(param.gamma == 0 && max_index > 0)
		param.gamma = 1.0/max_index;

	if(param.kernel_type == PRECOMPUTED)
		for(i=0;i<prob.l;i++)
		{
			if((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
			{
				mexPrintf("Wrong input format: sample_serial_number out of range\n");
				/*return -1;*/
			}
		}
    
	/*return 0;*/
}



/* -------------------------------------------------------------------
 Main MEX function - interface to Matlab.
create_kernel(yp,&prob, &param);
  [vec_x,exitflag,t,access,Nabla] = 
         gsmo_mex(X,y,kernel,f,a,b,LB,UB,x0,Nabla0,tmax,tolKKT,verb);

-------------------------------------------------------------------- */
void mexFunction( int nlhs, mxArray *plhs[],int nrhs, const mxArray*prhs[] )
{
  int verb;          
  uint32_t i;         
  uint32_t MaxIter;   
  double TolKKT; 
  double *vec_x;         /* output arg -- solution*/ 
  double *vec_x0;         
  double *diag_H;        /* diagonal of matrix H */
  double *f;             /* vector f */
  double *a;
  double b;
  double *LB;
  double *UB;
/*  double fval; */


  libqp_state_T state;
  const schar* y;
  
  /*------------------------------------------------------------------- */
  /* Take input arguments                                               */
  /*------------------------------------------------------------------- */
  if( nrhs != 15) mexErrMsgTxt("Incorrect number of input arguments.");

  /*mat_H = mxGetPr(prhs[0]); */
  nVar = mxGetM(prhs[0]);
  
  param.kernel_type = (int)(mxGetScalar(prhs[2]));  
  param.degree = (int)(mxGetScalar(prhs[3]));    
  param.gamma = (double)(mxGetScalar(prhs[4]));      
  param.coef0 = (double)(mxGetScalar(prhs[5]));
  
  /*
  mexPrintf("... %d\n", param.kernel_type );  
  mexPrintf("... %d\n", param.degree );
  mexPrintf("... %f\n", param.gamma );
  mexPrintf("... %f\n", param.coef0 );
  */
  
  f = mxGetPr(prhs[6]);   
  a = mxGetPr(prhs[7]);
  b = mxGetScalar(prhs[8]);
  LB = mxGetPr(prhs[9]);
  UB = mxGetPr(prhs[10]);
  vec_x0 = mxGetPr(prhs[11]);
  MaxIter = mxIsInf( mxGetScalar(prhs[12])) ? INT_MAX : (long)mxGetScalar(prhs[12]);
  TolKKT = mxGetScalar(prhs[13]); 
  verb = (int)(mxGetScalar(prhs[14])); 
  

  /* Read Problem */
  read_problem_dense(prhs[1], prhs[0]);
  
    
  if( verb > 0 ) {
    mexPrintf("Settings of QP solver\n");
    mexPrintf("MaxIter : %d\n", MaxIter );
    mexPrintf("TolKKT  : %f\n", TolKKT );
    mexPrintf("nVar    : %d\n", nVar );
    mexPrintf("verb    : %d\n", verb );
  }

  plhs[0] = mxCreateDoubleMatrix(nVar,1,mxREAL);
  vec_x = mxGetPr(plhs[0]);
  memcpy( vec_x, vec_x0, sizeof(double)*nVar );
 
  
  /*------------------------------------------------------------------- */
  /* Call QP solver                                                     */
  /*------------------------------------------------------------------- */
  state = gsmo_solver(&prob,&param,f,a,b,LB,UB,vec_x,nVar,MaxIter,TolKKT, NULL);

    
  /*------------------------------------------------------------------- */
  /* Generate outputs                                                   */
  /*------------------------------------------------------------------- */

  plhs[1] = mxCreateDoubleScalar(state.QP);
  plhs[2] = mxCreateDoubleScalar((double)state.exitflag);
  plhs[3] = mxCreateDoubleScalar((double)state.nIter);


  /*------------------------------------------------------------------- */
  /* Clean up                                                           */
  /*------------------------------------------------------------------- */
   free( prob.x );
   free( prob.y );
   free( x_space );
   
}

/* EOF */

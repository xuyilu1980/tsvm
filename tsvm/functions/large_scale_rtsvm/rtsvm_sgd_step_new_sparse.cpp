/*=================================================================
 * [W,W0] = tsvm_sgd_update( X, Y, idx, W,W0, stepSize, Cvec, mu, gamma, L, U, nUpdates, ConstraintFlag )
        
 *
 *=================================================================*/

#include <mex.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#define MAX(A,B) ((A) >= (B) ? (A) : (B))


void mexFunction( int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[] )
{

    if( nrhs != 14 )
        mexErrMsgTxt("Sixteen input arguments are required.\n\n"
                     "Synopsis:\n"
                     "  [W,W0] = tsvm_sgd_update( X, Y, idx, W,W0, stepSize, \n"
                     "                  Cvec, mu, gamma, L, U, nUpdates, ConstraintFlag) \n"
                     "   \n"
                     "\n");

    /* */   
    mwIndex* X_Jc      = (mwIndex*)mxGetJc( prhs[0] );
    double*  X_Pr      = (double*)mxGetPr( prhs[0] );
    mwIndex* X_Ir      = (mwIndex*)mxGetIr( prhs[0] );
    //  double *X          = (double*)mxGetPr( prhs[0] );
    double *Y          = (double*)mxGetPr( prhs[1] );
    double *idx        = (double*)mxGetPr( prhs[2] );
    double *W          = (double*)mxGetPr( prhs[3] );
    double W0          = (double)mxGetScalar( prhs[4] );
    double step_size   = (double)mxGetScalar( prhs[5] );
    double *Cvec       = (double*)mxGetPr( prhs[6] );
	double *mu         = (double*)mxGetPr( prhs[7] );
    double gamma       = (double)mxGetScalar( prhs[8] );
    double *betavec         = (double*)mxGetPr( prhs[9] );
    double L   = (double)mxGetScalar( prhs[10] );
	double U   = (double)mxGetScalar( prhs[11] );
    double n_updates   = (double)mxGetScalar( prhs[11] );
    double constraint_flag   = (double)mxGetScalar( prhs[13] );
    
    mwSize n_dims      = mxGetM(prhs[0]);
    mwSize n_examples  = mxGetN(prhs[0]);
    mwSize len_idx     = MAX( mxGetN(prhs[2]), mxGetM(prhs[2]));

    double norm_of_mu = 1.0;
    for( mwSize i = 0; i < n_dims; i++) norm_of_mu += mu[i]*mu[i];


    /*mexPrintf("n_examples    : %d\n", n_examples );
    mexPrintf("C        : %f %f \n", Cvec[0], Cvec[1] );
    mexPrintf("step_size     : %f\n", step_size );
    mexPrintf("n_dims        : %d\n", n_dims );
    mexPrintf("len_idx       : %d\n", len_idx );
    mexPrintf("norm_of_mu    : %f\n", norm_of_mu );
    mexPrintf("n_updates     : %.0f\n", n_updates );
    */

    plhs[0]            = mxCreateDoubleMatrix(n_dims, 1, mxREAL);
    double *new_W  = mxGetPr( plhs[0] );

    plhs[1]            = mxCreateDoubleMatrix(1, 1, mxREAL);
    double *new_W0 = mxGetPr( plhs[1] );

    memcpy( new_W, W, n_dims*sizeof( double ) );

    *new_W0     = W0;

    //mwSize num_iter_print = len_idx / 10;

    double const1 = step_size;
	double beta, CC, ri, ns;
	ns = L+U;
    
    for( mwSize i = 0; i < len_idx; i++ )
    {

      mwSize n     = (mwSize)idx[i] - 1;
	  ri= n+1;
	  if (ri<=L)
	  {
		  CC=Cvec[0];
	  }
	  else
	  { 
		  CC=Cvec[1];
	  }
      double label = Y[n];


      //      double* ptr_at_W = new_W;
      //double* ptr_at_X = X+n*n_dims;
      
      double score = *new_W0;
	  beta = betavec[n]; 
      
      mwIndex j    = X_Jc[n];
      mwSize  nne  = X_Jc[n+1] - X_Jc[n];  // number of non-zero elements in i-th column
      mwIndex ir;
      mwSize  k;


      for( k = 0; k < nne; k++ )
      {
        ir     = X_Ir[j];
        score += X_Pr[j]*new_W[ir];
        j++;
      }

      /*      for( mwSize j = 0; j < n_dims; j++ ) 
      {
        //        score += new_W[j]*X[n*n_dims+j];
        score += (*ptr_at_W) * (*ptr_at_X);
        ptr_at_W++;
        ptr_at_X++;
      }
      */
     

		
      score *= label;
      if( score < 1  && ri<=L )
      {
        *new_W0 -= (-label*CC*const1/ns)+(beta*label*const1/ns);
        
        //ptr_at_X = X+n*n_dims;
        //ptr_at_W = new_W;


        for( k = 0; k < n_dims; k++ )
        {
          new_W[k] -= const1* new_W[k]/ns;
        }        

        j = X_Jc[n];
        for( k = 0; k < nne; k++ )
        {
          ir         = X_Ir[j];
          new_W[ir] -= -label*const1*CC*X_Pr[j]/ns + label*const1*beta*X_Pr[j]/ns;
          j++;
        }

        /*        for( mwSize j = 0; j < n_dims; j++ )
        {
          new_W[j] -= -label*const1*CC*(*ptr_at_X)/ns + (label*const1*beta*(*ptr_at_X)/ns) + (const1* (*ptr_at_W)/ns);
          ptr_at_X++;
          ptr_at_W++;
          } */       
      }
      
	  else if( score >= 1  && ri<=L )
      {
        *new_W0 -= (beta*label*const1/ns);
        //        ptr_at_X = X+n*n_dims;
        //        ptr_at_W = new_W;

        for( k = 0; k < n_dims; k++ )
        {
          new_W[k] -= const1* new_W[k]/ns;
        }

        j = X_Jc[n];
        for( k = 0; k < nne; k++ )
        {
          ir         = X_Ir[j];
          new_W[ir] -= label*const1*beta*X_Pr[j]/ns;
          j++;
        }
        
        /*        for( mwSize j = 0; j < n_dims; j++ )
        {
          new_W[j] -= ( label*const1*beta*(*ptr_at_X)/ns) + (const1* (*ptr_at_W)/ns);
          ptr_at_X++;
          ptr_at_W++;
          }*/
        
      }

      else if( score < 1  && ri>L )
      {
        *new_W0 -= (-label*const1*CC/ns)+(beta*const1*label/ns);
        //        ptr_at_X = X+n*n_dims;
        //        ptr_at_W = new_W;

        for( k = 0; k < n_dims; k++ )
        {
          new_W[k] -= const1 * new_W[k] / ns;
        }

        j = X_Jc[n];
        for( k = 0; k < nne; k++ )
        {
          ir         = X_Ir[j];
          new_W[ir] -= -label*const1*CC*X_Pr[j]/ns + label*const1*beta*X_Pr[j]/ns;
          j++;
        }

        /*        for( mwSize j = 0; j < n_dims; j++ )
        {
          new_W[j] -= -label*const1*CC*(*ptr_at_X)/ns + (label*const1*beta*(*ptr_at_X)/ns) + (const1* (*ptr_at_W)/ns);
          ptr_at_X++;
          ptr_at_W++;
          }*/
        
      }


      else       
	  {
        *new_W0 -= (beta*label*const1/ns);
        //        ptr_at_X = X+n*n_dims;
        //        ptr_at_W = new_W;

        for( k = 0; k < n_dims; k++ )
        {
          new_W[k] -= const1* new_W[k] / ns;
        }

        j = X_Jc[n];
        for( k = 0; k < nne; k++ )
        {
          ir         = X_Ir[j];
          new_W[ir] -=  label*const1*beta*X_Pr[j]/ns;
          j++;
        }

        /*        for( mwSize j = 0; j < n_dims; j++ )
        {
          new_W[j] -= ( label*const1*beta*(*ptr_at_X)/ns) + (const1* (*ptr_at_W)/ns);
          ptr_at_X++;
          ptr_at_W++;
          }*/
        
      }

     
	  if (constraint_flag==1)
	  {
      
      // projection on balancing constr
      double sc = gamma - *new_W0;
      for( mwSize j = 0; j < n_dims; j++ ) sc -= mu[j]*new_W[j];

      sc /= norm_of_mu;

      *new_W0 += sc;
      for( mwSize j = 0; j < n_dims; j++ ) new_W[j] += sc*mu[j];
	  }
      
   
    //    mexPrintf("\n");
	  }

    return;
}

/*=================================================================
 * [W,W0] = tsvm_sgd_update( X, Y, idx, W,W0, stepSize, Cvec, mu, gamma, L, U, nUpdates, ConstraintFlag )
        
 *
 *=================================================================*/
//这应该是最关键的地方，如何通过梯度下降，使得目标函数值最小
#include <mex.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#define MAX(A,B) ((A) >= (B) ? (A) : (B))


void mexFunction( int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[] )
{

    if( nrhs != 14 )  //应该有14个参数
        mexErrMsgTxt("Sixteen input arguments are required.\n\n"
                     "Synopsis:\n"
                     "  [W,W0] = tsvm_sgd_update( X, Y, idx, W,W0, stepSize, \n"
                     "                  Cvec, mu, gamma, L, U, nUpdates, ConstraintFlag) \n"
                     "   \n"
                     "\n");

    /* */    
    double *X          = (double*)mxGetPr( prhs[0] ); //XX，样本，包含有标记和无标记的样本
    double *Y          = (double*)mxGetPr( prhs[1] );//标签
    double *idx        = (double*)mxGetPr( prhs[2] ); //打乱了样本索引
    double *W          = (double*)mxGetPr( prhs[3] ); //W,权
    double W0          = (double)mxGetScalar( prhs[4] );//偏离
    double step_size   = (double)mxGetScalar( prhs[5] );//步长，这里取了0.01
    double *Cvec       = (double*)mxGetPr( prhs[6] ); //带标签和不带标签的惩罚因子
	double *mu         = (double*)mxGetPr( prhs[7] );//就是原文中公式(14)的c值
    double gamma       = (double)mxGetScalar( prhs[8] );//有标签数据的y值的平均值
    double *betavec         = (double*)mxGetPr( prhs[9] );
    double L   = (double)mxGetScalar( prhs[10] );//有标签的数据总量
	double U   = (double)mxGetScalar( prhs[11] );//未标记的数据总量
    double n_updates   = (double)mxGetScalar( prhs[11] );//初始值为0
    double constraint_flag   = (double)mxGetScalar( prhs[13] );
    
    mwSize n_dims      = mxGetM(prhs[0]);//样本的维数
    mwSize n_examples  = mxGetN(prhs[0]);//所有样本的数量
    mwSize len_idx     = MAX( mxGetN(prhs[2]), mxGetM(prhs[2]));//L+2U的值

    double norm_of_mu = 1.0;
    for( mwSize i = 0; i < n_dims; i++) norm_of_mu += mu[i]*mu[i];//未标记样本的值的平方加1


    /*mexPrintf("n_examples    : %d\n", n_examples );
    mexPrintf("C        : %f %f \n", Cvec[0], Cvec[1] );
    mexPrintf("step_size     : %f\n", step_size );
    mexPrintf("n_dims        : %d\n", n_dims );
    mexPrintf("len_idx       : %d\n", len_idx );
    mexPrintf("norm_of_mu    : %f\n", norm_of_mu );
    mexPrintf("n_updates     : %.0f\n", n_updates );
    */

    plhs[0]            = mxCreateDoubleMatrix(n_dims, 1, mxREAL);
    double *new_W  = mxGetPr( plhs[0] );  //这个应该是新的W

    plhs[1]            = mxCreateDoubleMatrix(1, 1, mxREAL);
    double *new_W0 = mxGetPr( plhs[1] );  //这个应该是新的b

    memcpy( new_W, W, n_dims*sizeof( double ) );

    *new_W0     = W0;

    //mwSize num_iter_print = len_idx / 10;

    double const1 = step_size; //步长
	double beta, CC, ri, ns;
	ns = L+U; //所有的样本书，包括有标记的和未标记的
    
    for( mwSize i = 0; i < len_idx; i++ )
    {

      //      if( i > 0 & ((i % num_iter_print) == 0 ) | i == len_idx-1) 
      //      {
      //        mexPrintf("%.0f%% ", 100.0*(double)i/(double)(len_idx-1));
      //      }
      mwSize n     = (mwSize)idx[i] - 1;  //n=L+2U
	  ri= n+1;
	  if (ri<=L)
	  {
		  CC=Cvec[0]; //此时CC是带标签的惩罚因子
	  }
	  else
	  { 
		  CC=Cvec[1];//此时CC是无标签的惩罚因子
	  }
      double label = Y[n];


      double* ptr_at_W = new_W;//ptr_at_W指向权重
      double* ptr_at_X = X+n*n_dims;
      double score = *new_W0;//score等于偏移
	  beta = betavec[n]; 

      for( mwSize j = 0; j < n_dims; j++ ) 
      {
        //        score += new_W[j]*X[n*n_dims+j];
        score += (*ptr_at_W) * (*ptr_at_X);
        ptr_at_W++;
        ptr_at_X++;
      }
     

		
      score *= label;
      if( score < 1  && ri<=L )
      {
        *new_W0 -= (-label*CC*const1/ns)+(beta*label*const1/ns);//new_W0相当于bt，const1应该是相当于202页的lamdat
        ptr_at_X = X+n*n_dims;
        ptr_at_W = new_W;

        for( mwSize j = 0; j < n_dims; j++ )
        {
          //          new_W[j] -= -label*step_size*X[n*n_dims+j]/(double)len_idx + 
          //                      step_size*new_W[j]*lambda/(2*(double)len_idx);

          //          new_W[j] -= -label*const1*X[n*n_dims+j] + const2*new_W[j];
          new_W[j] -= -label*const1*CC*(*ptr_at_X)/ns + (label*const1*beta*(*ptr_at_X)/ns) + (const1* (*ptr_at_W)/ns);
          ptr_at_X++;
          ptr_at_W++;
        }
        
      }
      
	  else if( score >= 1  && ri<=L )
      {
        *new_W0 -= (beta*label*const1/ns);
        ptr_at_X = X+n*n_dims;
        ptr_at_W = new_W;

        for( mwSize j = 0; j < n_dims; j++ )
        {
          //          new_W[j] -= -label*step_size*X[n*n_dims+j]/(double)len_idx + 
          //                      step_size*new_W[j]*lambda/(2*(double)len_idx);

          //          new_W[j] -= -label*const1*X[n*n_dims+j] + const2*new_W[j];
          new_W[j] -= ( label*const1*beta*(*ptr_at_X)/ns) + (const1* (*ptr_at_W)/ns);
          ptr_at_X++;
          ptr_at_W++;
        }
        
      }

      else if( score < 1  && ri>L )
      {
        *new_W0 -= (-label*const1*CC/ns)+(beta*const1*label/ns);
        ptr_at_X = X+n*n_dims;
        ptr_at_W = new_W;

        for( mwSize j = 0; j < n_dims; j++ )
        {
          //          new_W[j] -= -label*step_size*X[n*n_dims+j]/(double)len_idx + 
          //                      step_size*new_W[j]*lambda/(2*(double)len_idx);

          //          new_W[j] -= -label*const1*X[n*n_dims+j] + const2*new_W[j];
          new_W[j] -= -label*const1*CC*(*ptr_at_X)/ns + (label*const1*beta*(*ptr_at_X)/ns) + (const1* (*ptr_at_W)/ns);
          ptr_at_X++;
          ptr_at_W++;
        }
        
      }


      else       
	  {
        *new_W0 -= (beta*label*const1/ns);
        ptr_at_X = X+n*n_dims;
        ptr_at_W = new_W;

        for( mwSize j = 0; j < n_dims; j++ )
        {
          //          new_W[j] -= -label*step_size*X[n*n_dims+j]/(double)len_idx + 
          //                      step_size*new_W[j]*lambda/(2*(double)len_idx);

          //          new_W[j] -= -label*const1*X[n*n_dims+j] + const2*new_W[j];
          new_W[j] -= ( label*const1*beta*(*ptr_at_X)/ns) + (const1* (*ptr_at_W)/ns);
          ptr_at_X++;
          ptr_at_W++;
        }
        
      }

     
	  if (constraint_flag==1)//满足平衡限制
	  {
      
      // projection on balancing constr
      double sc = gamma - *new_W0;  //相当于公式17中的r-cTv
      for( mwSize j = 0; j < n_dims; j++ ) sc -= mu[j]*new_W[j];

      sc /= norm_of_mu;//norm_of_mu相当于||C||

      *new_W0 += sc;
      for( mwSize j = 0; j < n_dims; j++ ) new_W[j] += sc*mu[j];
	  }
      
   
    //    mexPrintf("\n");
	  }//对应76行的括号，说明是对所有的样本进行如上处理,在这个大循环里它并没有使用break等类似语句中断循环
      //说明它是按部就班的来做这些事情的

    return;
}

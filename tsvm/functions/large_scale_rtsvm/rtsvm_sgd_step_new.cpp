/*=================================================================
 * [W,W0] = tsvm_sgd_update( X, Y, idx, W,W0, stepSize, Cvec, mu, gamma, L, U, nUpdates, ConstraintFlag )
        
 *
 *=================================================================*/
//��Ӧ������ؼ��ĵط������ͨ���ݶ��½���ʹ��Ŀ�꺯��ֵ��С
#include <mex.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#define MAX(A,B) ((A) >= (B) ? (A) : (B))


void mexFunction( int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[] )
{

    if( nrhs != 14 )  //Ӧ����14������
        mexErrMsgTxt("Sixteen input arguments are required.\n\n"
                     "Synopsis:\n"
                     "  [W,W0] = tsvm_sgd_update( X, Y, idx, W,W0, stepSize, \n"
                     "                  Cvec, mu, gamma, L, U, nUpdates, ConstraintFlag) \n"
                     "   \n"
                     "\n");

    /* */    
    double *X          = (double*)mxGetPr( prhs[0] ); //XX�������������б�Ǻ��ޱ�ǵ�����
    double *Y          = (double*)mxGetPr( prhs[1] );//��ǩ
    double *idx        = (double*)mxGetPr( prhs[2] ); //��������������
    double *W          = (double*)mxGetPr( prhs[3] ); //W,Ȩ
    double W0          = (double)mxGetScalar( prhs[4] );//ƫ��
    double step_size   = (double)mxGetScalar( prhs[5] );//����������ȡ��0.01
    double *Cvec       = (double*)mxGetPr( prhs[6] ); //����ǩ�Ͳ�����ǩ�ĳͷ�����
	double *mu         = (double*)mxGetPr( prhs[7] );//����ԭ���й�ʽ(14)��cֵ
    double gamma       = (double)mxGetScalar( prhs[8] );//�б�ǩ���ݵ�yֵ��ƽ��ֵ
    double *betavec         = (double*)mxGetPr( prhs[9] );
    double L   = (double)mxGetScalar( prhs[10] );//�б�ǩ����������
	double U   = (double)mxGetScalar( prhs[11] );//δ��ǵ���������
    double n_updates   = (double)mxGetScalar( prhs[11] );//��ʼֵΪ0
    double constraint_flag   = (double)mxGetScalar( prhs[13] );
    
    mwSize n_dims      = mxGetM(prhs[0]);//������ά��
    mwSize n_examples  = mxGetN(prhs[0]);//��������������
    mwSize len_idx     = MAX( mxGetN(prhs[2]), mxGetM(prhs[2]));//L+2U��ֵ

    double norm_of_mu = 1.0;
    for( mwSize i = 0; i < n_dims; i++) norm_of_mu += mu[i]*mu[i];//δ���������ֵ��ƽ����1


    /*mexPrintf("n_examples    : %d\n", n_examples );
    mexPrintf("C        : %f %f \n", Cvec[0], Cvec[1] );
    mexPrintf("step_size     : %f\n", step_size );
    mexPrintf("n_dims        : %d\n", n_dims );
    mexPrintf("len_idx       : %d\n", len_idx );
    mexPrintf("norm_of_mu    : %f\n", norm_of_mu );
    mexPrintf("n_updates     : %.0f\n", n_updates );
    */

    plhs[0]            = mxCreateDoubleMatrix(n_dims, 1, mxREAL);
    double *new_W  = mxGetPr( plhs[0] );  //���Ӧ�����µ�W

    plhs[1]            = mxCreateDoubleMatrix(1, 1, mxREAL);
    double *new_W0 = mxGetPr( plhs[1] );  //���Ӧ�����µ�b

    memcpy( new_W, W, n_dims*sizeof( double ) );

    *new_W0     = W0;

    //mwSize num_iter_print = len_idx / 10;

    double const1 = step_size; //����
	double beta, CC, ri, ns;
	ns = L+U; //���е������飬�����б�ǵĺ�δ��ǵ�
    
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
		  CC=Cvec[0]; //��ʱCC�Ǵ���ǩ�ĳͷ�����
	  }
	  else
	  { 
		  CC=Cvec[1];//��ʱCC���ޱ�ǩ�ĳͷ�����
	  }
      double label = Y[n];


      double* ptr_at_W = new_W;//ptr_at_Wָ��Ȩ��
      double* ptr_at_X = X+n*n_dims;
      double score = *new_W0;//score����ƫ��
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
        *new_W0 -= (-label*CC*const1/ns)+(beta*label*const1/ns);//new_W0�൱��bt��const1Ӧ�����൱��202ҳ��lamdat
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

     
	  if (constraint_flag==1)//����ƽ������
	  {
      
      // projection on balancing constr
      double sc = gamma - *new_W0;  //�൱�ڹ�ʽ17�е�r-cTv
      for( mwSize j = 0; j < n_dims; j++ ) sc -= mu[j]*new_W[j];

      sc /= norm_of_mu;//norm_of_mu�൱��||C||

      *new_W0 += sc;
      for( mwSize j = 0; j < n_dims; j++ ) new_W[j] += sc*mu[j];
	  }
      
   
    //    mexPrintf("\n");
	  }//��Ӧ76�е����ţ�˵���Ƕ����е������������ϴ���,�������ѭ��������û��ʹ��break����������ж�ѭ��
      //˵�����ǰ����Ͱ��������Щ�����

    return;
}

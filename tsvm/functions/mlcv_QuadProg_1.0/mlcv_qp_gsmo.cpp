/*-----------------------------------------------------------------------
 * libqp_gsmo.c: implementation of the Generalized SMO algorithm.
 *
 * DESCRIPTION
 *  The library provides function which solves the following instance of
 *  a convex Quadratic Programming task:
 *
 *  min QP(x) := 0.5*x'*H*x + f'*x  
 *   x                                      
 *
 *   s.t.    a'*x = b 
 *           LB[i] <= x[i] <= UB[i]   for all i=1..n
 *
 * A precision of the found solution is controlled by the input argument
 * TolKKT which defines tightness of the relaxed Karush-Kuhn-Tucker 
 * stopping conditions.
 *
 * INPUT ARGUMENTS
 *  get_col   function which returns pointer to the i-th column of H.
 *  diag_H [double n x 1] vector containing values on the diagonal of H.
 *  f [double n x 1] vector.
 *  a [double n x 1] Vector which must not contain zero entries.
 *  b [double 1 x 1] Scalar.
 *  LB [double n x 1] Lower bound; -inf is allowed.
 *  UB [double n x 1] Upper bound; inf is allowed.
 *  x [double n x 1] solution vector; must be feasible.
 *  n [uint32_t 1 x 1] dimension of H.
 *  MaxIter [uint32_t 1 x 1] max number of iterations.
 *  TolKKT [double 1 x 1] Tightness of KKT stopping conditions.
 *  print_state  print function; if == NULL it is not called.
 *
 * RETURN VALUE
 *  structure [libqp_state_T]
 *   .QP [1x1] Primal objective value.
 *   .exitflag [1 x 1] Indicates which stopping condition was used:
 *     -1  ... not enough memory
 *      0  ... Maximal number of iterations reached: nIter >= MaxIter.
 *      4  ... Relaxed KKT conditions satisfied. 
 *   .nIter [1x1] Number of iterations.
 *
 * REFERENCE
 *  S.S. Keerthi, E.G. Gilbert. Convergence of a generalized SMO algorithm 
 *   for SVM classier design. Technical Report CD-00-01, Control Division, 
 *   Dept. of Mechanical and Production Engineering, National University 
 *   of Singapore, 2000. 
 *   http://citeseer.ist.psu.edu/keerthi00convergence.html  
 *
 *
 * Copyright (C) 2006-2008 Vojtech Franc, xfrancv@cmp.felk.cvut.cz
 * Center for Machine Perception, CTU FEL Prague
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public 
 * License as published by the Free Software Foundation; 
 * Version 3, 29 June 2007
 *-------------------------------------------------------------------- */

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>

#include "mlcv_qp.h"

#include "mex.h"

#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif
template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
template <class S, class T> static inline void clone(T*& dst, S* src, int n)
{
	dst = new T[n];
	memcpy((void *)dst,(void *)src,sizeof(T)*n);
}
static inline double powi(double base, int times)
{
	double tmp = base, ret = 1.0;

	for(int t=times; t>0; t/=2)
	{
		if(t%2==1) ret*=tmp;
		tmp = tmp * tmp;
	}
	return ret;
}
#define INF HUGE_VAL
#define TAU 1e-12
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))



//
// Kernel Cache
//
// l is the number of total data items
// size is the cache size limit in bytes
//
class Cache
{
public:
	Cache(int l,long int size);
	~Cache();

	// request data [0,len)
	// return some position p where [p,len) need to be filled
	// (p >= len if nothing needs to be filled)
	int get_data(const int index, Qfloat **data, int len);
	void swap_index(int i, int j);	
private:
	int l;
	long int size;
	struct head_t
	{
		head_t *prev, *next;	// a circular list
		Qfloat *data;
		int len;		// data[0,len) is cached in this entry
	};

	head_t *head;
	head_t lru_head;
	void lru_delete(head_t *h);
	void lru_insert(head_t *h);
};

Cache::Cache(int l_,long int size_):l(l_),size(size_)
{
	head = (head_t *)calloc(l,sizeof(head_t));	// initialized to 0
	size /= sizeof(Qfloat);
	size -= l * sizeof(head_t) / sizeof(Qfloat);
	size = max(size, 2 * (long int) l);	// cache must be large enough for two columns
	lru_head.next = lru_head.prev = &lru_head;
}

Cache::~Cache()
{
	for(head_t *h = lru_head.next; h != &lru_head; h=h->next)
		free(h->data);
	free(head);
}

void Cache::lru_delete(head_t *h)
{
	// delete from current location
	h->prev->next = h->next;
	h->next->prev = h->prev;
}

void Cache::lru_insert(head_t *h)
{
	// insert to last position
	h->next = &lru_head;
	h->prev = lru_head.prev;
	h->prev->next = h;
	h->next->prev = h;
}

int Cache::get_data(const int index, Qfloat **data, int len)
{
	head_t *h = &head[index];
	if(h->len) lru_delete(h);
	int more = len - h->len;

	if(more > 0)
	{
		// free old space
		while(size < more)
		{
			head_t *old = lru_head.next;
			lru_delete(old);
			free(old->data);
			size += old->len;
			old->data = 0;
			old->len = 0;
		}

		// allocate new space
		h->data = (Qfloat *)realloc(h->data,sizeof(Qfloat)*len);
		size -= more;
		swap(h->len,len);
	}

	lru_insert(h);
	*data = h->data;
	return len;
}

void Cache::swap_index(int i, int j)
{
	if(i==j) return;

	if(head[i].len) lru_delete(&head[i]);
	if(head[j].len) lru_delete(&head[j]);
	swap(head[i].data,head[j].data);
	swap(head[i].len,head[j].len);
	if(head[i].len) lru_insert(&head[i]);
	if(head[j].len) lru_insert(&head[j]);

	if(i>j) swap(i,j);
	for(head_t *h = lru_head.next; h!=&lru_head; h=h->next)
	{
		if(h->len > i)
		{
			if(h->len > j)
				swap(h->data[i],h->data[j]);
			else
			{
				// give up
				lru_delete(h);
				free(h->data);
				size += h->len;
				h->data = 0;
				h->len = 0;
			}
		}
	}
}

//
// Kernel evaluation
//
// the static method k_function is for doing single kernel evaluation
// the constructor of Kernel prepares to calculate the l*l kernel matrix
// the member function get_Q is for getting one column from the Q Matrix
//
class QMatrix {
public:
	virtual Qfloat *get_Q(int column, int len) const = 0;
	virtual Qfloat *get_QD() const = 0;
	virtual void swap_index(int i, int j) const = 0;
	virtual ~QMatrix() {}
};

class Kernel: public QMatrix {
public:
	Kernel(int l, svm_node * const * x, const svm_parameter& param);
	virtual ~Kernel();

	static double k_function(const svm_node *x, const svm_node *y,
				 const svm_parameter& param);
	virtual Qfloat *get_Q(int column, int len) const = 0;
	virtual Qfloat *get_QD() const = 0;
	virtual void swap_index(int i, int j) const	// no so const...
	{
		swap(x[i],x[j]);
		if(x_square) swap(x_square[i],x_square[j]);
	}
protected:

	double (Kernel::*kernel_function)(int i, int j) const;

private:
	const svm_node **x;
	double *x_square;

	// svm_parameter
	const int kernel_type;
	const int degree;
	const double gamma;
	const double coef0;

	static double dot(const svm_node *px, const svm_node *py);
    static double sum_min( const svm_node *px, const svm_node *py );
	static double chi_square( const svm_node *px, const svm_node *py );
	
	double kernel_linear(int i, int j) const
	{
		return dot(x[i],x[j]);
	}
	double kernel_poly(int i, int j) const
	{
		return powi(gamma*dot(x[i],x[j])+coef0,degree);
	}
	double kernel_rbf(int i, int j) const
	{
		return exp(-gamma*(x_square[i]+x_square[j]-2*dot(x[i],x[j])));
	}
	double kernel_sigmoid(int i, int j) const
	{
		return tanh(gamma*dot(x[i],x[j])+coef0);
	}
    double kernel_summin(int i, int j) const
	{
		return sum_min( x[i], x[j] );
	}
	double kernel_chisquare(int i, int j) const
	{
		return chi_square( x[i], x[j] );
	}
	double kernel_rbf_chisquare(int i, int j) const
	{       
		return exp(-gamma*(chi_square(x[i],x[j]))); 
	}
	/*double kernel_precomputed(int i, int j) const
	{
		return x[i][(int)(x[j][0].value)].value;
	}*/
};


Kernel::Kernel(int l, svm_node * const * x_, const svm_parameter& param)
:kernel_type(param.kernel_type), degree(param.degree),
 gamma(param.gamma), coef0(param.coef0)
{
	switch(kernel_type)
	{
		case LINEAR:
			kernel_function = &Kernel::kernel_linear;
			break;
		case POLY:
			kernel_function = &Kernel::kernel_poly;
			break;
		case RBF:
			kernel_function = &Kernel::kernel_rbf;
			break;
		case SIGMOID:
			kernel_function = &Kernel::kernel_sigmoid;
			break;
        case SUMMIN:
			kernel_function = &Kernel::kernel_summin;
			break;
		case CHISQUARE:
			kernel_function = &Kernel::kernel_chisquare;
			break;
		case RBF_CHISQUARE:
			kernel_function = &Kernel::kernel_rbf_chisquare;
			break;
		/*case PRECOMPUTED:
			kernel_function = &Kernel::kernel_precomputed;
			break;
         */
	}

	clone(x,x_,l);

	if(kernel_type == RBF)
	{
		x_square = new double[l];
		for(int i=0;i<l;i++)
			x_square[i] = dot(x[i],x[i]);
	}
	else
		x_square = 0;
}

Kernel::~Kernel()
{
	delete[] x;
	delete[] x_square;
}

double Kernel::dot(const svm_node *px, const svm_node *py)
{
	double sum = 0;
	while(px->index != -1 && py->index != -1)
	{
		if(px->index == py->index)
		{
			sum += px->value * py->value;
			++px;
			++py;
		}
		else
		{
			if(px->index > py->index)
				++py;
			else
				++px;
		}			
	}
	return sum;
}

double Kernel::sum_min( const svm_node *px, const svm_node *py)
{
	double sum = 0;

	while( px->index != -1 && py->index != -1 )
	{
		if ( px->index == py->index )
		{
			sum += min( px->value, py->value );
			++px;
			++py;
		}
		else
		{
			if ( px->index > py->index )
				++py;
			else
				++px;
		}
	}

	return sum;
} // end Kernel::sum_min()



double Kernel::chi_square( const svm_node *px, const svm_node *py)
{    
	double dr = 0;
	double sr = 0;
	double sum = 0;

	while ( px->index != -1 && py->index != -1 )
	{
		if (px->index==py->index)
		{
			dr = py->value - px->value;
			sr = py->value + px->value;

			if (dr != 0 && sr > 0)
			{
				sum += (dr * dr) / (2 * sr);
			}

			++px;
			++py;

		}
		else
		{
			if ( px->index > py->index )
				++py;
			else
				++px;
		}	
	}

	return sum;

} // end Kernel::chi_square()


//
// Q matrices for various formulations
//
class SVC_Q: public Kernel
{ 
public:
	SVC_Q(const svm_problem& prob, const svm_parameter& param, const schar *y_)
	:Kernel(prob.l, prob.x, param)
	{
		clone(y,y_,prob.l);
		cache = new Cache(prob.l,(long int)(param.cache_size*(1<<20)));
		QD = new Qfloat[prob.l];
		for(int i=0;i<prob.l;i++)
		{
			QD[i]= (Qfloat)(this->*kernel_function)(i,i);
			//mexPrintf("QD : %4.2f\n",QD[i]);
		}
	}
 	
	Qfloat *get_Q(int i, int len) const
	{
		Qfloat *data;
		int start, j;
		if((start = cache->get_data(i,&data,len)) < len)
		{
			for(j=start;j<len;j++)
			{
				data[j] = (Qfloat)(y[i]*y[j]*(this->*kernel_function)(i,j));
			    //mexPrintf("data : %4.2f\n",data[j]);
			}

		}
		return data;
	}

	Qfloat *get_QD() const
	{
		return QD;
	}

	void swap_index(int i, int j) const
	{
		cache->swap_index(i,j);
		Kernel::swap_index(i,j);
		swap(y[i],y[j]);
		swap(QD[i],QD[j]);
	}

	~SVC_Q()
	{
		delete[] y;
		delete cache;
		delete[] QD;
	}
private:
	schar *y;
	Cache *cache;
	Qfloat *QD;
};


libqp_state_T gsmo_solver(const svm_problem* prob, 
                        const svm_parameter* param,
                        double *f,
                        double *a,
                        double b,
                        double *LB,
                        double *UB,
                        double *x,
                        uint32_t n,
                        uint32_t MaxIter,
                        double TolKKT,
                        void (*print_state)(libqp_state_T state))       
{

    int l = prob->l;
	double *minus_ones = new double[l];
	schar *y = new schar[l];
    

	for(int k=0;k<l;k++)
	{
		//alpha[i] = 0;
		minus_ones[k] = -1;
		if(prob->y[k] > 0) y[k] = +1; else y[k]=-1;
	}
        
    const QMatrix& Q = SVC_Q(*prob,*param,y);
    
    const Qfloat* QD = Q.get_QD();
    
    //for(int k=0;k<l;k++)
    //    mexPrintf("k %d : % f \n",k,QD[k]);
    
    const Qfloat* Q_col_u;
    const Qfloat* Q_col_v;
    
    /*  
    mexPrintf("%d\n",param->kernel_type);
    mexPrintf("%d\n",param->degree);
    mexPrintf("%d\n",param->gamma);
    mexPrintf("%d\n",param->coef0);
    */
    
    //double *col_u;
    //double *col_v;
    double *Nabla;
    double minF_up;
    double maxF_low;
    double tau;
    double F_i;
    double tau_ub, tau_lb;
    //double Q_P;
    uint32_t i, j;
    uint32_t u, v;
    libqp_state_T state;

    Nabla = NULL;

    /* ------------------------------------------------------------ */
    /* Initialization                                               */
    /* ------------------------------------------------------------ */

    /* Nabla = H*x + f is gradient*/
    Nabla = (double*)LIBQP_CALLOC(n, sizeof(double));
        
    //mexPrintf("n : %d\n",n);
    
    if( Nabla == NULL )
    {
      state.exitflag=-1;
      goto cleanup;
    }

    /* compute gradient */
    for( i=0; i < n; i++ ) 
    { 
      
        Nabla[i] += f[i];
        
        if( x[i] != 0 ) {

            Q_col_u = Q.get_Q(i,l);            
            
            for( j=0; j < n; j++ ) {
                
                Nabla[j] += Q_col_u[j]*x[i];  
                
            }
        }
    }

    
    if( print_state != NULL) 
    {
     state.QP = 0;
     for(i = 0; i < n; i++ ) 
        state.QP += 0.5*(x[i]*Nabla[i]+x[i]*f[i]); 

     print_state( state );
    }

        
    /* ------------------------------------------------------------ */
    /* Main optimization loop                                       */
    /* ------------------------------------------------------------ */

    state.nIter = 0;
    state.exitflag = 100;
    while( state.exitflag == 100 ) 
    {
        state.nIter ++;   
        
        //mexPrintf("%d\n",state.nIter);

        /* find the most violating pair of variables */
        minF_up = LIBQP_PLUS_INF;
        maxF_low = -LIBQP_PLUS_INF;
        
        for(i = 0; i < n; i++ ) 
        {
            F_i = Nabla[i]/a[i];

          if(LB[i] < x[i] && x[i] < UB[i]) 
          { /* i is from I_0 */
            
                if( minF_up > F_i) { minF_up = F_i; u = i; }
                if( maxF_low < F_i) { maxF_low = F_i; v = i; }
          } 
          else if((a[i] > 0 && x[i] == LB[i]) || (a[i] < 0 && x[i] == UB[i])) 
          { /* i is from I_1 or I_2 */
                if( minF_up > F_i) { minF_up = F_i; u = i; }
          }
          else if((a[i] > 0 && x[i] == UB[i]) || (a[i] < 0 && x[i] == LB[i])) 
          { /* i is from I_3 or I_4 */

                if( maxF_low < F_i) { maxF_low = F_i; v = i; }
          }
        
        }
        

    /* check KKT conditions */
    if( maxF_low - minF_up <= TolKKT )
      state.exitflag = 4;
    else 
    {

      /* SMO update of the most violating pair */
        Q_col_u = Q.get_Q(u,l);
        Q_col_v = Q.get_Q(v,l);

      if( a[u] > 0 ) 
         { tau_lb = (LB[u]-x[u])*a[u]; tau_ub = (UB[u]-x[u])*a[u]; }
      else
         { tau_ub = (LB[u]-x[u])*a[u]; tau_lb = (UB[u]-x[u])*a[u]; }

      if( a[v] > 0 )
         { tau_lb = LIBQP_MAX(tau_lb,(x[v]-UB[v])*a[v]); tau_ub = LIBQP_MIN(tau_ub,(x[v]-LB[v])*a[v]); }
      else
         { tau_lb = LIBQP_MAX(tau_lb,(x[v]-LB[v])*a[v]); tau_ub = LIBQP_MIN(tau_ub,(x[v]-UB[v])*a[v]); }

      
      tau = (Nabla[v]/a[v]-Nabla[u]/a[u])/
            (QD[u]/(a[u]*a[u]) + QD[v]/(a[v]*a[v]) - 2*Q_col_u[v]/(a[u]*a[v]));
            

      tau = LIBQP_MIN(LIBQP_MAX(tau,tau_lb),tau_ub);

      x[u] += tau/a[u];
      x[v] -= tau/a[v];

      /* update Nabla */
      for(i = 0; i < n; i++ ) 
         Nabla[i] += Q_col_u[i]*tau/a[u] - Q_col_v[i]*tau/a[v];

    }

    if( state.nIter >= MaxIter )
      state.exitflag = 0;

    if( print_state != NULL) 
    {
      state.QP = 0;
      for(i = 0; i < n; i++ ) 
        state.QP += 0.5*(x[i]*Nabla[i]+x[i]*f[i]); 

      print_state( state );
    }

    }  

    /* compute primal objective value */
    state.QP = 0;
    for(i = 0; i < n; i++ ) 
        state.QP += 0.5*(x[i]*Nabla[i]+x[i]*f[i]); 

    cleanup:  
 
        /* clear resources */
        LIBQP_FREE(Nabla);
    
        delete [] minus_ones;
        delete [] y;
    
        return( state ); 


    }

   
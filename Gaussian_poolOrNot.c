/*
 *  Author: Paul Horton
 *  Copyright: Paul Horton 2021, All rights reserved.
 *  Created: 20211201
 *  Updated: 20260513
 *  Licence: GPLv3
 *  Description: Simple demonstration of a Bayesian way to guess at the number of components
 *               behind a sample of numerical data.
 *  Compile:  gcc -Wall -O3 -o Gaussian_poolOrNot Gaussian_poolOrNot.c GSLfun.c timespec_utils.c -lgsl -lgslcblas -lm
 *  Environment: $GSL_RNG_SEED
 */
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include "GSLfun.h"
#include "timespec_utils.h"
/* ───────────  Global definitions and variables  ────────── */
#define DATA_N 50
#define CDF_GAUSS_N 20
#define CDF_GAMMA_N 10
#define CDF_JBETA_N 40

// Block of code used twice in main()
#define COMPUTE_DATA_PROBS_PRINT_RESULTS(NUM)\
  prob_data1_bySampling=  data_prob_1component_bySampling();               \
  prob_data2_bySampling=  data_prob_2component_bySampling();               \
  prob_data1_bySumming =  data_prob_1component_bySumming();                \
  prob_data2_bySumming =  data_prob_2component_bySumming();                \
  if(  prob_data1_bySampling > prob_data2_bySampling  )   ++model##NUM##_sampling_favors1; \
  if(  prob_data1_bySumming  > prob_data2_bySumming   )   ++model##NUM##_summing__favors1; \
  printf( "Data maximum likelihood under (one,two) component model= (%g,TODO)\n", \
          data_Gauss1_maxLikelihood() , data_Gauss2_maxLikelihood()  \
          );                                                               \
  printf( "Integrals by sampling= (%g,%g)  by summing: (%g,%g)\n\n",       \
          prob_data1_bySampling, prob_data2_bySampling,                    \
          prob_data1_bySumming, prob_data2_bySumming )



#define ADD_TIME_AFTER_BEFORE(METHOD) \
  timespec_setToNow( &after_time );      \
  tot_msec_time[METHOD]  += timespec_diff_msecs( before_time, after_time )


typedef struct{
  double mixCof;
  Gauss_params Gauss1;
  Gauss_params Gauss2;
} Gauss_mixture_params;


const Gauss_params mu_prior_params= {0.0, 4.0};
const double sigma_prior_param_a= 0.5;
const double sigma_prior_param_b= 2.0;


double data[DATA_N];
const uint dataN= DATA_N;

const uint sampleRepeatNum= 2000000;


// To record computation times in milliseconds for each step or method you wish to record.
double tot_msec_time[5];

// Names for positions in array `tot_msec_time'.
// "T_" prefix used here mainly to reduce the risk of name conflicts.
enum {SUM_PRECOMP, T_SUM1, T_SAMP1, T_SUM2, T_SAMP2};


// Vars (re)used by functions below to hold times.
struct timespec before_time;
struct timespec after_time;

////////////////////////取樣功能/////////////////////////
FILE* sample_file= NULL;
char* cur_dataset_source="";
int dataset_idx=0;

void sampling_write(const char* model, int iteration, double total){

  if(!sample_file){ 
    printf("no file found.\n");
    return;
    }


  int trace_num=sampleRepeatNum/100; // 100圈紀錄一次
  if(iteration%trace_num == 0 || iteration == trace_num ){
    fprintf(sample_file, "%s,%d,%g\n", model, iteration, total/(double)iteration);
  }
  else{
    return;
  }
  
}
///////////////////////////////////////////////////////////

/* ───────────  Functions to help summarize or dump the data  ────────── */

double data_sample_mean(){
  double mean= 0.0;
  for( uint i= 0;  i < dataN;  ++i ){
    mean += data[i];
  }
  return  mean / (double) dataN;
}

double data_sample_variance(){
  double mean= data_sample_mean();
  double var= 0.0;
  for( uint i= 0;  i < dataN;  ++i ){
    double diff=  data[i] - mean;
    var +=  diff * diff;
  }
  return  var / (double) (dataN);
}



// ternary CMP function for use with qsort
int CMPdata( const void *arg1, const void *arg2 ){
  return(
         (*(double*) arg1 < *(double*) arg2)? -1 :
         (*(double*) arg2 < *(double*) arg1)? +1 :
         /* else    *arg1 == *arg2   */        0);
}

void data_print(){
  qsort(  data,  dataN,  sizeof(double), CMPdata  );
  for( uint i= 0;  i < dataN;  ++i ){
    printf( "%u\t+%5.3f ", i, data[i] );
  }
}


/* ───────────  Functions used for sampling/generating data   ────────── */

Gauss_params prior_Gauss_params_sample(){
  Gauss_params params;
  params.mu=   GSLfun_ran_gaussian( mu_prior_params );
  params.sigma=  sigma_of_precision( GSLfun_ran_gamma(sigma_prior_param_a, sigma_prior_param_b) );
  return  params;
}


Gauss_mixture_params prior_Gauss_mixture_params_sample(){

  // You can adjust these parameters, but cdfInv_precompute assumes a=b
  static const double betaDist_a= 0.75;
  static const double betaDist_b= 0.75;

  Gauss_mixture_params params;
  params.mixCof=  GSLfun_ran_beta( betaDist_a, betaDist_b );
  params.Gauss1=  prior_Gauss_params_sample();
  params.Gauss2=  prior_Gauss_params_sample();
  return  params;
}


void data_generate_1component( Gauss_params params ){
  for( uint i= 0; i < dataN; ++i ){
    data[i]=  GSLfun_ran_gaussian( params );
  }
}

void data_generate_2component( Gauss_mixture_params params ){
  for( uint i= 0; i < dataN; ++i ){
    data[i]=  GSLfun_ran_gaussian
      (gsl_ran_flat01() < params.mixCof?  params.Gauss1  : params.Gauss2);
  }
}


/* ───────────  Numerical integration precomputation  ────────── */

// Arrays to hold precomputed values.
double cdfInv_Gauss[CDF_GAUSS_N];  const double cdf_Gauss_n= CDF_GAUSS_N;
double cdfInv_gamma[CDF_GAMMA_N];  const double cdf_gamma_n= CDF_GAMMA_N;
double cdfInv_JBeta[CDF_JBETA_N];  const double cdf_JBeta_n= CDF_JBETA_N;

//  Precompute the cumulative probabilities of μ and σ discrete values.
//  The probabilities depend on the current prior_params values
void cdfInv_precompute(){
  double x;

  timespec_setToNow( &before_time );

  // Since Normal range is unbounded, precompute cdfInv for vals:  ¹⁄₍ₙ₊₁₎...ⁿ⁄₍ₙ₊₁₎
  for(  uint i= 0; i < cdf_Gauss_n; ++i  ){
    x= (i+1) / (double) (1+cdf_Gauss_n);
    cdfInv_Gauss[i]=  gsl_cdf_gaussian_Pinv( x, mu_prior_params.sigma );
  }
  for(  uint i= 0; i < cdf_gamma_n; ++i  ){
    x= i / (double) (cdf_gamma_n);
    cdfInv_gamma[i]=  gsl_cdf_gamma_Pinv( x, sigma_prior_param_a, sigma_prior_param_b );
    //printf( "cdfInv_Gamma[%u]= %g\n", i, cdfInv_gamma[i] );
  }
  for(  uint i= 0; i < cdf_JBeta_n; ++i  ){
    // By symmetry, only need Beta values for p ≦ 0.5.  For example p=0.8, is the same p=0.2 with Gauss components swapped.
    x= 0.5 * i / (double) (cdf_JBeta_n);
    cdfInv_JBeta[i]=  gsl_cdf_beta_Pinv( x, 0.5, 0.5 );
    //printf( "cdfInv_JBeta[%u]= %g\n", i, cdfInv_JBeta[i] );
  }

  ADD_TIME_AFTER_BEFORE( SUM_PRECOMP );
}




/* ───────────  Probability Computations on the Data  ────────── */

// Return Ｐ[D|μ,σ]
double prob_data_given_1Gauss( const Gauss_params params ){
  double prob= 1.0;
  for(  uint d= 0;  d < dataN;  ++d  ){
    prob *= GSLfun_ran_gaussian_pdf( data[d], params );
  }
  return prob;
}


// Return Ｐ[D|m,μ₁,σ₁,μ₂,σ₂]
double prob_data_given_2Gauss( const double mixCof, const Gauss_params Gauss1, const Gauss_params Gauss2  ){
    double prob= 1.0;
    for( uint i= 0; i < dataN; ++i ){
      prob *=   (1-mixCof) * GSLfun_ran_gaussian_pdf( data[i], Gauss2 )
              +    mixCof  * GSLfun_ran_gaussian_pdf( data[i], Gauss1 );
    }
    return prob;
}

// Add for computing log likelihood (2 components)
double log_prob_data_given_2Gauss(const double mixCof, const Gauss_params Gauss1, const Gauss_params Gauss2){
  double logprob = 0.0;
  for( uint i = 0; i < dataN; ++i ){
    double p1 = mixCof * GSLfun_ran_gaussian_pdf( data[i], Gauss1 );
    double p2 = (1.0 - mixCof) * GSLfun_ran_gaussian_pdf( data[i], Gauss2 );
    logprob += log( p1 + p2 );
  }
  return logprob;
}

/* Return maximum likelihood of the data using a single Gaussian
 *
 * max_{μ,σ} Ｐ[D|μ,σ}
*/
double data_Gauss1_maxLikelihood(){
  Gauss_params params=  { data_sample_mean(), sqrt( data_sample_variance() ) };
  return prob_data_given_1Gauss( params );
}


/* Todo:
 *   Implement a maximum likelihood estimation routine for the two component model.
 *   Could use the soft k-means approach of the D. MacKay book Chapter 22.
 *   I use GMM-EM algorithm
 *
 *   double data_Gauss2_maxLikelihood(){...}
*/
double data_Gauss2_maxLikelihood(){

  const uint maxIter = 500;
  const double tol = 1e-10; // tolerance for convergence, based on log likelihood change
  const double minSigma = 1e-6;
  const double minWeight = 1e-6;

  // Initialization
  double mean = data_sample_mean();
  double std = sqrt( data_sample_variance() );

  if( std < minSigma ){
    std = 1.0;
  }

  Gauss_params g1 = { mean - 0.5 * std, std };
  Gauss_params g2 = { mean + 0.5 * std, std };
  double mixCof = 0.5;

  double prevLogLik = -INFINITY; // previos LogLik
  double curLogLik = log_prob_data_given_2Gauss( mixCof, g1, g2 ); // current LogLik

  // if( trace_EM ){
  //   printf("\n[EM init]\n");
  //   printf("iter=%3d  logLik=% .10f  mix=% .6f  "
  //          "g1=(mu=% .6f, sigma=% .6f)  "
  //          "g2=(mu=% .6f, sigma=% .6f)\n",
  //          -1, curLogLik, mixCof,
  //          g1.mu, g1.sigma,
  //          g2.mu, g2.sigma);
  // }

  // GMM-EM algorithm
  for( uint iter = 0; iter < maxIter; ++iter ){

    double N1 = 0.0; //effective number of points assigned to component 1
    double N2 = 0.0; //effective number of points assigned to component 1
    double sum1 = 0.0;
    double sum2 = 0.0;

    // E-step + collect sums for mean update
    for( uint i = 0; i < dataN; ++i ){

      double p1 = mixCof * GSLfun_ran_gaussian_pdf( data[i], g1 );
      double p2 = (1.0 - mixCof) * GSLfun_ran_gaussian_pdf( data[i], g2 );
      double denom = p1 + p2;

      double r1;
      if( denom <= 0.0 || !isfinite(denom) ){
        r1 = 0.5;
      }else{
        r1 = p1 / denom;
      }

      double r2 = 1.0 - r1;

      N1 += r1;
      N2 += r2;

      sum1 += r1 * data[i];
      sum2 += r2 * data[i];
    }

    // check if N1 or N2 is too small, to avoid one of the components collapsing to zero weight
    if( N1 < minWeight * dataN || N2 < minWeight * dataN ){
      // if( trace_EM ){
      //   printf("[EM stop] dead component: N1=%g, N2=%g\n", N1, N2);
      // }
      break;
    }

    // M step: update means
    double new_mu1 = sum1 / N1;
    double new_mu2 = sum2 / N2;

    // M step: update variances
    double var1 = 0.0;
    double var2 = 0.0;

    for( uint i = 0; i < dataN; ++i ){

      double p1 = mixCof * GSLfun_ran_gaussian_pdf( data[i], g1 );
      double p2 = (1.0 - mixCof) * GSLfun_ran_gaussian_pdf( data[i], g2 );
      double denom = p1 + p2;

      double r1;
      if( denom <= 0.0 || !isfinite(denom) ){
        r1 = 0.5;
      }else{
        r1 = p1 / denom;
      }

      double r2 = 1.0 - r1;

      double diff1 = data[i] - new_mu1;
      double diff2 = data[i] - new_mu2;

      var1 += r1 * diff1 * diff1;
      var2 += r2 * diff2 * diff2;
    }

    var1 /= N1;
    var2 /= N2;

    if( var1 < minSigma * minSigma ){
      var1 = minSigma * minSigma;
    }

    if( var2 < minSigma * minSigma ){
      var2 = minSigma * minSigma;
    }

    // M step update
    mixCof = N1 / (double)dataN;
    g1.mu = new_mu1;
    g2.mu = new_mu2;
    g1.sigma = sqrt(var1);
    g2.sigma = sqrt(var2);

    // Check convergence by log likelihood
    prevLogLik = curLogLik;
    curLogLik = log_prob_data_given_2Gauss( mixCof, g1, g2 );

    // if( trace_EM ){
    //   printf("iter=%3u  logLik=% .10f  diff=% .3e  mix=% .6f  "
    //          "N1=% .4f  N2=% .4f  "
    //          "g1=(mu=% .6f, sigma=% .6f)  "
    //          "g2=(mu=% .6f, sigma=% .6f)\n",
    //          iter, curLogLik, curLogLik - prevLogLik, mixCof,
    //          N1, N2,
    //          g1.mu, g1.sigma,
    //          g2.mu, g2.sigma);
    // }

    if( curLogLik + 1e-8 < prevLogLik ){
      printf("[WARNING] EM log-likelihood decreased: prev=%g, cur=%g\n",
             prevLogLik, curLogLik);
    }

    if( fabs(curLogLik - prevLogLik) < tol ){
      if( trace_EM ){
        printf("[EM stop] converged at iter=%u\n", iter);
      }
      break;
    }
  }

  // Return likelihood
  curLogLik = log_prob_data_given_2Gauss( mixCof, g1, g2 );
  return exp(curLogLik);
  // return curLogLik;
}



/* Compute Riemann sum to approximate the integral
 *
 * ∫ μ,σ  Ｐ[D|μ,σ]
 *
*/
double data_prob_1component_bySumming(){
  double prob_total= 0.0;

  timespec_setToNow( &before_time );
  for(  uint m= 0;  m < cdf_Gauss_n;  ++m  ){
    double mu= cdfInv_Gauss[m];
    for(  uint s= 0;  s < cdf_gamma_n;  ++s  ){
      double sigma=  sigma_of_precision( cdfInv_gamma[s] );
      Gauss_params cur_params= {mu, sigma};
      prob_total += prob_data_given_1Gauss( cur_params );
    }
  }

  ADD_TIME_AFTER_BEFORE( T_SUM1 );
  return  prob_total / (double) (cdf_Gauss_n * cdf_gamma_n);
}


/* Compute Riemann sum to approximate integral
 *
 * ∫ m,μ₁,σ₁,μ₂,σ₂  P[D|m,μ₁,σ₁,μ₂,σ₂]
 *
*/
double data_prob_2component_bySumming(){
  double prob_total= 0.0;

  timespec_setToNow( &before_time );
  for(  uint m1= 0;  m1 < cdf_Gauss_n;  ++m1  ){
    double mu1= cdfInv_Gauss[m1];
    for(  uint m2= 0;  m2 < cdf_Gauss_n;  ++m2  ){
      double mu2= cdfInv_Gauss[m2];
      for(  uint s1= 0;  s1 < cdf_gamma_n;  ++s1  ){
        double sigma1=  sigma_of_precision( cdfInv_gamma[s1] );
        Gauss_params cur_params1= {mu1, sigma1};
        for(  uint s2= 0;  s2 < cdf_gamma_n;  ++s2  ){
          double sigma2=  sigma_of_precision( cdfInv_gamma[s2] );
          Gauss_params cur_params2= {mu2, sigma2};
          for(  uint mi= 0;  mi < cdf_JBeta_n;  ++mi  ){
            double mixCof= cdfInv_JBeta[mi];
            prob_total +=  prob_data_given_2Gauss( mixCof, cur_params1, cur_params2 );
          }
        }
      }
    }
  }

  ADD_TIME_AFTER_BEFORE( T_SUM2 );
  return  prob_total / (double) (cdf_Gauss_n * cdf_Gauss_n * cdf_gamma_n * cdf_gamma_n * cdf_JBeta_n);
}



/*  Use sampling to estimate
 *  ∫ μ,σ  P[D|μ,σ]
 */
double data_prob_1component_bySampling(){
  double prob_total= 0.0;

  timespec_setToNow( &before_time );
  for( uint iter= 0;  iter < sampleRepeatNum; ++iter ){
    Gauss_params params= prior_Gauss_params_sample();
    prob_total += prob_data_given_1Gauss( params );
    sampling_write("Model1", iter+1, prob_total);
  }

  ADD_TIME_AFTER_BEFORE( T_SAMP1 );
  return  prob_total / (double) sampleRepeatNum;
}



/*  Use sampling to estimate
 *  ∫ m,μ₁,σ₁,μ₂,σ₂  P[D|m,μ₁,σ₁,μ₂,σ₂]
 */
double data_prob_2component_bySampling(){
  double prob_total= 0.0;

  timespec_setToNow( &before_time );
  for( uint iter= 0;  iter < sampleRepeatNum; ++iter ){
    Gauss_mixture_params params=  prior_Gauss_mixture_params_sample();
    prob_total += prob_data_given_2Gauss( params.mixCof, params.Gauss1, params.Gauss2 );
    sampling_write("Model2", iter+1, prob_total);
  }

  ADD_TIME_AFTER_BEFORE( T_SAMP2 );
  return  prob_total / (double) sampleRepeatNum;
}




// Is flag NAME given on command line?
// NAME includes leading hyphens, e.g. '--help'
bool cliFlagGivenP( int* argcR, char *argv[], const char* name ){
  for( int i= 0; i < *argcR; ++i ){
    // If NAME found splice it out and decrement ARGC
    if( !strcmp( argv[i], name ) ){
      for( int j= i; j < *argcR-1; ++j ){
         argv[j]= argv[j+1];
      }
      --(*argcR);
      return true;
    }
  }
  return false;
}


int main( int argc, char *argv[] ){

  uint datasets_n= 10;

  // So we can skip the 1-component generated datasets when we want to.
  bool skip_1component_datasetsP= cliFlagGivenP( &argc, argv, "--skip1c" );
  
  {
    char usage_fmt[]=  "Usage: %s [--skip1c] [num_datasets]\n";
    switch( argc ){
    case 1:
      break;
    case 2:
      datasets_n=  atoi( argv[1] );
      if( !datasets_n ){
        printf(  usage_fmt, argv[0]  );
        exit( 64 );
      }
      break;
    default:
      printf(  usage_fmt, argv[0]  );
      exit( 64 );
    }
  }

  GSLfun_setup();
  double prob_data1_bySampling, prob_data2_bySampling;
  double prob_data1_bySumming,  prob_data2_bySumming;
  /////////取樣設定////////////////////////////////////////////
  sample_file= fopen("sample_integral_trace.csv", "w");
  fprintf(sample_file, "model,iteration,average_prob\n");
  /////////////////////////////////////////////////////////////

  cdfInv_precompute();


  uint model1_sampling_favors1=  0;
  uint model1_summing__favors1=  0;
  uint model2_sampling_favors1=  0;
  uint model2_summing__favors1=  0;



  printf(  "Starting computation for %d datasets each. ...\n",  datasets_n  );

  if( !skip_1component_datasetsP ){
    printf( "\nData generated with one component\n" );
    for(  uint iter= 0;  iter < datasets_n;  ++iter  ){
      Gauss_params model_params = prior_Gauss_params_sample();
      printf(  "generating data with: (μ,σ) =  (%4.2f,%4.2f)\n", model_params.mu, model_params.sigma  );
      data_generate_1component( model_params );
      COMPUTE_DATA_PROBS_PRINT_RESULTS(1);
    }
  }

  printf( "\nData generated with two components\n" );
  for(  uint iter= 0;  iter < datasets_n;  ++iter  ){
    Gauss_mixture_params model_params=  prior_Gauss_mixture_params_sample();
    printf(  "generating data with:  m; (μ1,σ1); (μ2,σ2) =  %5.3f; (%4.2f,%4.2f); (%4.2f,%4.2f)\n",
             model_params.mixCof,
             model_params.Gauss1.mu, model_params.Gauss1.sigma,
             model_params.Gauss2.mu, model_params.Gauss2.sigma  );
    data_generate_2component( model_params );
    COMPUTE_DATA_PROBS_PRINT_RESULTS(2);
  }

  printf( "By sampling: Model1 data, correct selection %u/%u\n", model1_sampling_favors1, datasets_n  );
  printf( "             Model2 data, correct selection %u/%u\n", (datasets_n - model2_sampling_favors1), datasets_n  );
  printf( "By summing:  Model1 data, correct selection %u/%u\n", model1_summing__favors1, datasets_n  );
  printf( "             Model2 data, correct selection %u/%u\n", (datasets_n - model2_summing__favors1), datasets_n  );

  printf( "total times spent are:\n-PRECOMP-\t--SUM1---\t--SAMP1--\t--SUM2---\t--SAMP2-\n" );
  for( int i= 0; i < sizeof(tot_msec_time)/sizeof(tot_msec_time[0]); i++ ){
    if( i )  printf( "\t" );
    printf( "%9g", tot_msec_time[i] );
  }
  printf( "\n" );

  if( sample_file ){
    fclose( sample_file );
    printf( "Sample output written to sample_integral_trace.csv\n" );
  }

}

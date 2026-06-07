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
#include <stdlib.h>
/* ───────────  Global definitions and variables  ────────── */
#define DATA_N 50
#define CDF_GAUSS_N 20 //systematic summing 時，μ 用 20 個 percentile 點
#define CDF_GAMMA_N 10 //precision prior 用 10 個 percentile 點
#define CDF_JBETA_N 40 //mixture coefficient b 用 40 個點

// EM add
static bool trace_EM = false;

// Block of code used twice in main()
#define COMPUTE_DATA_PROBS_PRINT_RESULTS(NUM)\
  prob_data1_bySampling=  data_prob_1component_bySampling();               \
  prob_data2_bySampling=  data_prob_2component_bySampling();               \
  prob_data1_bySumming =  data_prob_1component_bySumming();                \
  prob_data2_bySumming =  data_prob_2component_bySumming();                \
  if(  prob_data1_bySampling > prob_data2_bySampling  )   ++model##NUM##_sampling_favors1; \
  if(  prob_data1_bySumming  > prob_data2_bySumming   )   ++model##NUM##_summing__favors1; \     
  printf( "Data maximum log-likelihood under (one,two) component model= (%g,%g)\n", \
        data_Gauss1_maxLogLikelihood(), \
        data_Gauss2_maxLogLikelihood() \
      );                                                         



#define ADD_TIME_AFTER_BEFORE(METHOD) \
  timespec_setToNow( &after_time );      \
  tot_msec_time[METHOD]  += timespec_diff_msecs( before_time, after_time )


typedef struct{
  double mixCof;
  Gauss_params Gauss1;
  Gauss_params Gauss2;
} Gauss_mixture_params;


const Gauss_params mu_prior_params= {0.0, 4.0}; //{mu, sigma}
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

// Add for computing log likelihood (1 component)
double log_prob_data_given_1Gauss( const Gauss_params params ){
  double logprob = 0.0;
  for( uint i = 0; i < dataN; ++i ){
    double p = GSLfun_ran_gaussian_pdf( data[i], params );
    logprob += log(p);
  }
  return logprob;
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
double data_Gauss1_maxLogLikelihood(){
  Gauss_params params=  { data_sample_mean(), sqrt( data_sample_variance() ) };
  // return prob_data_given_1Gauss( params );
  return log_prob_data_given_1Gauss( params );
}


/* Todo:
 *   Implement a maximum likelihood estimation routine for the two component model.
 *   Could use the soft k-means approach of the D. MacKay book Chapter 22.
 *   I use GMM-EM algorithm
 *
 *   double data_Gauss2_maxLikelihood(){...}
*/
double data_Gauss2_maxLogLikelihood(){

  const uint maxIter = 500;
  const double tol = 1e-10; // tolerance for convergence, based on log likelihood change

  // Initialization
  double mean = data_sample_mean();
  double std = sqrt( data_sample_variance() );

  Gauss_params g1 = { mean - 0.5 * std, std };
  Gauss_params g2 = { mean + 0.5 * std, std };
  double mixCof = 0.5;

  double prevLogLik = -INFINITY; // previos LogLik
  double curLogLik = log_prob_data_given_2Gauss( mixCof, g1, g2 ); // current LogLik

  if( trace_EM ){
    printf("\n[EM init]\n");
    printf("iter=%3d  logLik=% .10f  mix=% .6f  "
           "g1=(mu=% .6f, sigma=% .6f)  "
           "g2=(mu=% .6f, sigma=% .6f)\n",
           -1, curLogLik, mixCof,
           g1.mu, g1.sigma,
           g2.mu, g2.sigma);
  }

  // GMM-EM algorithm
  for( uint iter = 0; iter < maxIter; ++iter ){

    double N1 = 0.0; //effective number of points assigned to component 1
    double N2 = 0.0; //effective number of points assigned to component 2
    double sum1 = 0.0;
    double sum2 = 0.0;

    // E-step + collect sums for mean update
    for( uint i = 0; i < dataN; ++i ){

      double p1 = mixCof * GSLfun_ran_gaussian_pdf( data[i], g1 );
      double p2 = (1.0 - mixCof) * GSLfun_ran_gaussian_pdf( data[i], g2 );
      double denom = p1 + p2;

      double r1;
      r1 = p1 / denom;

      double r2 = 1.0 - r1;

      N1 += r1;
      N2 += r2;

      sum1 += r1 * data[i];
      sum2 += r2 * data[i];
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
      r1 = p1 / denom;

      double r2 = 1.0 - r1;

      double diff1 = data[i] - new_mu1;
      double diff2 = data[i] - new_mu2;

      var1 += r1 * diff1 * diff1;
      var2 += r2 * diff2 * diff2;
    }

    var1 /= N1;
    var2 /= N2;

    // M step update
    mixCof = N1 / (double)dataN;
    g1.mu = new_mu1;
    g2.mu = new_mu2;
    g1.sigma = sqrt(var1);
    g2.sigma = sqrt(var2);

    // Check convergence by log likelihood
    prevLogLik = curLogLik;
    curLogLik = log_prob_data_given_2Gauss( mixCof, g1, g2 );

    if( trace_EM ){
      printf("iter=%3u  logLik=% .10f  diff=% .3e  mix=% .6f  "
             "N1=% .4f  N2=% .4f  "
             "g1=(mu=% .6f, sigma=% .6f)  "
             "g2=(mu=% .6f, sigma=% .6f)\n",
             iter, curLogLik, curLogLik - prevLogLik, mixCof,
             N1, N2,
             g1.mu, g1.sigma,
             g2.mu, g2.sigma);
    }

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
  // curLogLik = log_prob_data_given_2Gauss( mixCof, g1, g2 );
  // return exp(curLogLik);
  return curLogLik;
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

// test func
void run_EM_test(){

  /*
   * A simple artificial two-component dataset.
   *
   * True model:
   *   component 1: N(-3, 0.5)
   *   component 2: N(+3, 0.5)
   */
  Gauss_mixture_params true_params;
  true_params.mixCof = 0.5;
  true_params.Gauss1.mu = -3.0;
  true_params.Gauss1.sigma = 0.5;
  true_params.Gauss2.mu = 3.0;
  true_params.Gauss2.sigma = 0.5;

  data_generate_2component( true_params );

  printf("\n========== EM test ==========\n");

  printf("True generating model:\n");
  printf("  mixCof = %.6f\n", true_params.mixCof);
  printf("  Gauss1 = N(mu=%.6f, sigma=%.6f)\n",
         true_params.Gauss1.mu, true_params.Gauss1.sigma);
  printf("  Gauss2 = N(mu=%.6f, sigma=%.6f)\n",
         true_params.Gauss2.mu, true_params.Gauss2.sigma);

  printf("\nGenerated data summary:\n");
  printf("  dataN = %u\n", dataN);
  printf("  sample mean = %.10f\n", data_sample_mean());
  printf("  sample variance = %.10f\n", data_sample_variance());
  printf("  sample std = %.10f\n", sqrt(data_sample_variance()));

  double ll1 = data_Gauss1_maxLogLikelihood();
  double ll2 = data_Gauss2_maxLogLikelihood();

  printf("\nMaximum log-likelihood comparison:\n");
  printf("  one-component max log-likelihood = %.10f\n", ll1);
  printf("  two-component EM log-likelihood  = %.10f\n", ll2);
  printf("  difference ll2 - ll1             = %.10f\n", ll2 - ll1);

  if( ll2 >= ll1 ){
    printf("\ntwo-component model has higher or equal maximum log-likelihood.\n");
  }else{
    printf("\ntwo-component model is lower. EM may be stuck in a local optimum.\n");
  }

  printf("==================================\n\n");
}

int main( int argc, char *argv[] ){

  trace_EM = cliFlagGivenP( &argc, argv, "--traceEM" );

  GSLfun_setup();

  run_EM_test();

  return 0;
}
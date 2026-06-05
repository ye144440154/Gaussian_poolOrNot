#include "GSLfun.h"

static gsl_rng* gslRNG;


void Gauss_params_print( Gauss_params params ){
  printf(  "(%+4.2f,%4.2f)", params.mu, params.sigma  );
}

void GSLfun_setup(){
  if(  !getenv( "GSL_RNG_SEED" )  )   printf(  "Using default random seed\n" );
  gsl_rng_env_setup();
  gslRNG= gsl_rng_alloc(gsl_rng_mt19937);
}


double GSLfun_ran_beta( double a, double b ){
  return  gsl_ran_beta( gslRNG, a, b );
}

double GSLfun_ran_beta_Jeffreys(){
  return  gsl_ran_beta( gslRNG, 0.5, 0.5 );
}

uint   GSLfun_ran_binomial( double p, uint n ){
  return  gsl_ran_binomial( gslRNG, p, n );
}

double GSLfun_ran_gamma( double a, double theta ){
  return  gsl_ran_gamma( gslRNG, a, theta );
}

double GSLfun_ran_gaussian( Gauss_params params ){
  return  params.mu + gsl_ran_gaussian( gslRNG, params.sigma );
}

double GSLfun_ran_gaussian_pdf( double x, Gauss_params params  ){
  return  gsl_ran_gaussian_pdf( x-params.mu, params.sigma );
}

double gsl_ran_flat01(){
  return  gsl_ran_flat( gslRNG, 0.0, 1.0 );
}


double sigma_of_precision( double precision ){
  return  sqrt( 1.0 / precision );
}

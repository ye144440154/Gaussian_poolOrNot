#pragma once
#include <math.h>
#include <stdio.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_gamma.h>


typedef  unsigned int  uint;

typedef struct{
  double mu;
  double sigma;
} Gauss_params;

void Gauss_params_print( Gauss_params params );

void GSLfun_setup();

double GSLfun_ran_beta( double a, double b );
double GSLfun_ran_beta_Jeffreys();
uint   GSLfun_ran_binomial( double p, uint n );
double GSLfun_ran_gamma( double a, double theta );
double GSLfun_ran_gaussian( Gauss_params params );
double GSLfun_ran_gaussian_pdf( double x, Gauss_params params );

double gsl_ran_flat01();


double sigma_of_precision( double precision );

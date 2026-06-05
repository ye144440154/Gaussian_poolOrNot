#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "GSLfun.h"
/*
 *  Example program showing how to use GSL to sample from a beta distribution.
 *
 *  Compile:  gcc -o BetaBinomial_Jeffreys_sample  BetaBinomial_Jeffreys_sample.c  GSLfun.c -lgsl -lgslcblas -lm
 *  Environment: $GSL_RNG_SEED
 */


int main( int argc, char *argv[] ){

  GSLfun_setup();

  for( int i = 0; i < 10; i++ ){
    printf(  "Sampling from Beta(0.5,0.5), drew r=%g\n",
             GSLfun_ran_beta_Jeffreys()
             );
  }
}

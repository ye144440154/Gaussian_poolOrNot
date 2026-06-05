#include "timespec_utils.h"


// (Re)set *timevalR to the current time.
void timespec_setToNow( struct timespec* timevalR ){
  if(  !timespec_get( timevalR, TIME_UTC )  ){
    fprintf( stderr, "Failed to get the time.\n" );
    exit( EXIT_FAILURE );
  }
}


// Return time difference  AFTER - BEFORE  in milliseconds
double timespec_diff_msecs( const struct timespec before, const struct timespec after ){
  return
    1000 * (after.tv_sec  - before.tv_sec)  +
    1E-6 * (after.tv_nsec - before.tv_nsec);
}

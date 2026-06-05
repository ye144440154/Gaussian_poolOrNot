#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// (Re)set *timevalR to the current time.
void timespec_setToNow( struct timespec* timevalR );


// Return time difference  AFTER - BEFORE  in milliseconds
double timespec_diff_msecs( const struct timespec before, const struct timespec after );


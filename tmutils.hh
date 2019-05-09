#ifndef __MAKEMORE_TMUTILS__
#define __MAKEMORE_TMUTILS__ 1

#include <assert.h>
#include <sys/time.h>

inline double now() {
  struct timeval tv;
  assert(0 == gettimeofday(&tv, NULL));
  return ((double)tv.tv_sec + (double)tv.tv_usec / 1000000.0);
}

#endif


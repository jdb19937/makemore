#ifndef __MAKEMORE_RANDOM_HH__
#define __MAKEMORE_RANDOM_HH__

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <sys/types.h>

inline double rnd() {
  double r = (double)(rand() % (1 << 24)) / (double)(1 << 24);
  return r;
}

inline double rnd(double a, double b) {
  double r = a + rnd() * (b - a);
  return r;
}

#endif

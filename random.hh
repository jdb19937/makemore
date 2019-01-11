#ifndef __MAKEMORE_RANDOM_HH__
#define __MAKEMORE_RANDOM_HH__

#include <math.h>
#include <assert.h>

extern void seedrand();
extern void seedrand(unsigned int n);

extern double randgauss();
extern double randrange(double a, double b);
inline double randunit() {
  return randrange(0.0, 1.0);
}
extern unsigned int randuint();

inline double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
inline double unsigmoid(double x) {
  assert(x > 0 && x < 1);
  return log(-(x/(x-1)));
}

#endif

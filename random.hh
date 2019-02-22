#ifndef __MAKEMORE_RANDOM_HH__
#define __MAKEMORE_RANDOM_HH__

#include <assert.h>
#include <stdint.h>

#include <math.h>

namespace makemore {

extern void seedrand();
extern void seedrand(unsigned int n);

extern double randexp(double lambda = 1.0);
extern double randgauss();
extern double randrange(double a, double b);
extern unsigned int randuint();

inline uint8_t randbit() { return randuint() % 2; }

inline double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
inline double unsigmoid(double x) {
  assert(x > 0 && x < 1);
  return log(-(x/(x-1)));
}

extern uint64_t hash64(const uint8_t *data, unsigned int len);

}

#endif

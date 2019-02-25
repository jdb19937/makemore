#ifndef __MAKEMORE_NUMUTILS_HH__
#define __MAKEMORE_NUMUTILS_HH__ 1

#include <stdint.h>
#include <math.h>

namespace makemore {

inline double btod(uint8_t b) {
  return ((double)b / 256.0);
}

inline uint8_t dtob(double d) {
  d *= 256.0;
  if (d > 255.0)
    d = 255.0;
  if (d < 0.0)
    d = 0.0;
  return ((uint8_t)(d + 0.5));
}

inline void btodv(const uint8_t *b, double *d, unsigned int n) {
  for (unsigned int i = 0; i < n; ++i)
    d[i] = btod(b[i]);
}

inline void dtobv(const double *d, uint8_t *b, unsigned int n) {
  for (unsigned int i = 0; i < n; ++i)
    b[i] = dtob(d[i]);
}

}

#endif

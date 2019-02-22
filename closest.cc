#include <stdio.h>
#include <assert.h>

#include <vector>
#include <algorithm>

#include "closest.hh"

namespace makemore {

unsigned int closest(const double *x, const double *m, unsigned int k, unsigned int n) {
  assert(n > 0);
  assert(k > 0);

  double bestz = -1;
  unsigned int besti = 0;

  for (unsigned int i = 0; i < n; ++i) {
    const double *y = m + i * k;
 
    double z = 0;
    for (unsigned int j = 0; j < k; ++j) {
      double d = x[j] - y[j];
      z += d * d;
    }

    if (bestz < 0 || z < bestz) {
      bestz = z;
      besti = i;
    }
 fprintf(stderr, "closest i=%u z=%lf besti=%u bestz=%lf\n", i, z, besti, bestz);
  }
fprintf(stderr, "best i=%u\n", besti);

  return besti;
}

unsigned int maxdot(const double *x, const double *m, unsigned int k, unsigned int n) {
  assert(n > 0);
  assert(k > 0);

  double bestz = -1;
  unsigned int besti = 0;

  for (unsigned int i = 0; i < n; ++i) {
    const double *y = m + i * k;
 
    double z = 0;
    for (unsigned int j = 0; j < k; ++j) {
      z += x[j] * y[j];
    }

fprintf(stderr, "maxdot i=%u z=%lf\n", i, z);
    if (bestz < 0 || z > bestz) {
      bestz = z;
      besti = i;
    }
  }
fprintf(stderr, "best i=%u\n", besti);

  return besti;
}

}

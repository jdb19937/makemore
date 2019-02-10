#include <vector>
#include <algorithm>

#include <assert.h>

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
  }

  return besti;
}

}

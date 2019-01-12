#include <vector>
#include <algorithm>

#include <assert.h>


unsigned int closest(const double *x, const double *m, unsigned int k, unsigned int n) {
  assert(n > 0);
  assert(k > 0);
  std::vector<std::pair<double, unsigned int>> distind;

  distind.resize(n);
  for (unsigned int i = 0; i < n; ++i) {
    const double *y = m + i * k;
 
    double z = 0;
    for (unsigned int j = 0; j < k; ++j)
      z += (x[j] - y[j]) * (x[j] - y[j]);
    distind[i] = std::make_pair(z, i);
  }

  std::sort(distind.begin(), distind.end());
  return distind.begin()->second;
}

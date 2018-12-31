#define __MAKEMORE_TWIDDLE_CC__ 1
#include <assert.h>

#include "twiddle.hh"

void untwiddle3(const double *lo, const double *hi, unsigned int w, unsigned int h, double *z) {
  assert(w % 2 == 0 && h % 2 == 0);
  unsigned int nw = w / 2;
  unsigned int nh = h / 2;

  unsigned int ilo = 0, ihi = 0;

  unsigned int w3 = w * 3;
  for (unsigned int y = 0; y < h; y += 2) {
    for (unsigned int x = 0; x < w; x += 2) {
      for (unsigned int c = 0; c < 3; ++c) {
        unsigned int p = y * w3 + x * 3 + c;

        double m = lo[ilo++];
        double l = (hi[ihi++] - 0.5) * 2.0;
        double t = (hi[ihi++] - 0.5) * 2.0;
        double s = (hi[ihi++] - 0.5) * 2.0;

        z[p] = m + l + t + s;
        z[p+3] = m - l + t - s;
        z[p+w3] = m + l - t - s;
        z[p+w3+3] = m - l - t + s;
      }
    }
  }
}

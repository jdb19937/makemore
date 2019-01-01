#define __MAKEMORE_TWIDDLE_CC__ 1
#include <assert.h>

#include "twiddle.hh"

void untwiddle1(const double *lo, const double *hi, unsigned int w, unsigned int h, double *z) {
  assert(w % 2 == 0 && h % 2 == 0);
  unsigned int nw = w / 2;
  unsigned int nh = h / 2;

  unsigned int ilo = 0, ihi = 0;

  for (unsigned int y = 0; y < h; y += 2) {
    for (unsigned int x = 0; x < w; x += 2) {
      unsigned int p = y * w + x;

      double m = lo[ilo++];
      double l = (hi[ihi++] - 0.5) * 2.0;
      double t = (hi[ihi++] - 0.5) * 2.0;
      double s = (hi[ihi++] - 0.5) * 2.0;

      z[p] = m + l + t + s;
      z[p+1] = m - l + t - s;
      z[p+w] = m + l - t - s;
      z[p+w+1] = m - l - t + s;
    }
  }
}

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

void twiddle1(const double *z, unsigned int w, unsigned int h, double *lo, double *hi) {
  assert(w % 2 == 0);
  assert(h % 2 == 0);

  unsigned int nw = w / 2;
  unsigned int nh = h / 2;

  unsigned int ilo = 0, ihi = 0;
  for (unsigned int y = 0; y < h; y += 2) {
    for (unsigned int x = 0; x < w; x += 2) {
      unsigned int p = y * w + x;

      double m = (z[p] + z[p + 1] + z[p + w] + z[p + w + 1]) / 4.0;
      double l = (z[p] + z[p + w]) / 2.0 - m;
      double t = (z[p] + z[p + 1]) / 2.0 - m;
      double s = (z[p] + z[p + w + 1]) / 2.0 - m;

      lo[ilo++] = m;
      hi[ihi++] = 0.5 + l / 2.0;
      hi[ihi++] = 0.5 + t / 2.0;
      hi[ihi++] = 0.5 + s / 2.0;
    }
  }
}

void twiddle3(const double *z, unsigned int w, unsigned int h, double *lo, double *hi) {
  assert(w % 2 == 0);
  assert(h % 2 == 0);

  unsigned int nw = w / 2;
  unsigned int nh = h / 2;

  unsigned int ilo = 0, ihi = 0;
  unsigned int w3 = w * 3;
  for (unsigned int y = 0; y < h; y += 2) {
    for (unsigned int x = 0; x < w; x += 2) {
      for (unsigned int c = 0; c < 3; ++c) {
        unsigned int p = y * w3 + x * 3 + c;

        double m = (z[p] + z[p + 3] + z[p + w3] + z[p + w3 + 3]) / 4.0;
        double l = (z[p] + z[p + w3]) / 2.0 - m;
        double t = (z[p] + z[p + 3]) / 2.0 - m;
        double s = (z[p] + z[p + w3 + 3]) / 2.0 - m;

        lo[ilo++] = m;
        hi[ihi++] = 0.5 + l / 2.0;
        hi[ihi++] = 0.5 + t / 2.0;
        hi[ihi++] = 0.5 + s / 2.0;
      }
    }
  }
}

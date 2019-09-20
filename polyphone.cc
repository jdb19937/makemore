#define __MAKEMORE_POLYPHONE_CC__
#include "polyphone.hh"
#include "cudamem.hh"

#include <assert.h>
#include <string.h>
#include <math.h>

#include <vector>
#include <algorithm>
#include <map>

namespace makemore {

void Polyphone::from_au(const int8_t *au) {
  double *row = tab;
  double *tmp = new double[w];

  for (unsigned int y = 0; y < h; ++y) {

    for (unsigned int i = 0; i < w; ++i) {
      tmp[i] = (double)au[i] / 128.0;
    }
    dctii(tmp, w, tmp);

    std::multimap<double, unsigned int> vi;
    for (unsigned int i = 0; i < w; ++i) {
      vi.insert(std::make_pair(fabs(tmp[i]), i));
    }

    assert(c < w);
    auto vii = vi.rbegin();
    for (unsigned int j = 0; j < c; ++j) {
      unsigned int i = vii->second;
      double v = tmp[i];
      row[0] = (double)i / (double)w;
      row[1] = v;
      ++vii;
      row += 2;
    }

    au += w;
  }

  delete[] tmp;
}

void Polyphone::to_au(int8_t *au) {
  double *row = tab;
  double *tmp = new double[w];

  for (unsigned int y = 0; y < h; ++y) {
    memset(tmp, 0, sizeof(double) * w);

    for (unsigned int j = 0; j < c; ++j) {
      double v = row[1];
      int i = (int)(row[0] * (double)w);
      if (i >= w) i = w - 1;
      if (i < 0) i = 0;
      tmp[i] = v;
      row += 2;
    }

    dctiii(tmp, w, tmp);

    for (unsigned int i = 0; i < w; ++i) {
      double x = tmp[i] * 128.0;
      if (x < -127) x = -127;
      if (x > 127) x = 127;
      au[i] = (int8_t)x;
    }

    au += w;
  }

  delete[] tmp;
}

}

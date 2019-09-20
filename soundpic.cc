#define __MAKEMORE_SOUNDPIC_CC__
#include "soundpic.hh"
#include "cudamem.hh"

#include <assert.h>
#include <math.h>

#include <vector>
#include <algorithm>

namespace makemore {

void Soundpic::from_au(const int8_t *au) {
  double *row = tab;

  for (unsigned int y = 0; y < h; ++y) {
    for (unsigned int i = 0; i < w; ++i) {
      row[i] = (double)au[i] / 128.0;
    }
    dctii(row, w, row);

    for (unsigned int i = 0; i < w; ++i) {
      row[i] *= (4.0 * (double)(i + 1) / (double)w);
    }

    row += w;
    au += w;
  }
}

void Soundpic::to_au(int8_t *au) {
  double *row = tab;
  double *tmp = new double[w];

  for (unsigned int y = 0; y < h; ++y) {
    for (unsigned int i = 0; i < w; ++i) {
      row[i] /= (4.0 * (double)(i + 1) / (double)w);
    }

    dctiii(row, w, tmp);
    for (unsigned int i = 0; i < w; ++i) {
      double x = tmp[i] * 128.0;
      if (x < -127) x = -127;
      if (x > 127) x = 127;
      au[i] = (int8_t)x;
    }

    au += w;
    row += w;
  }

  delete[] tmp;
}

void Soundpic::from_rgb(const uint8_t *rgb) {
  double *row = tab;

  for (unsigned int y = 0; y < h; ++y) {
    for (unsigned int i = 0; i < w; ++i) {
      double x = (rgb[0] + rgb[1] + rgb[2]) / 3.0;
      row[i] = ((x - 128) / 128.0) / 2.0;
      rgb += 3;
    }

    row += w;
  }
}


void Soundpic::to_rgb(uint8_t *rgb) {
  double *row = tab;

  for (unsigned int y = 0; y < h; ++y) {
    for (unsigned int i = 0; i < w; ++i) {
      double x = 128 + (row[i] * 128.0 * 2.0);
      if (x < 0) x = 0;
      if (x > 255.0) x = 255.0;
      rgb[0] = rgb[1] = rgb[2] = (uint8_t)x;
      rgb += 3;
    }

    row += w;
  }
}

void Soundpic::mask(double level) {
  double *row = tab;

  std::vector<double> tmp;
  tmp.resize(w);

  for (unsigned int y = 0; y < h; ++y) {
    for (unsigned int i = 0; i < w; ++i)
      tmp[i] = fabs(row[i]);
    std::sort(tmp.begin(), tmp.end());

    unsigned int k = (unsigned int)((double)w * level);
    assert(k < w);
    double min = tmp[k];

    for (unsigned int i = 0; i < w; ++i)
      if (row[i] > -min && row[i] < min)
        row[i] = 0;

    row += w;
  }
}

}

#ifndef __MAKEMORE_POLYPHONE_HH__
#define __MAKEMORE_POLYPHONE_HH__ 1

#include <stdint.h>

namespace makemore {

struct Polyphone {
  unsigned int w, h, c;
  double *tab;

  Polyphone(unsigned int _w, unsigned int _h, unsigned int _c) : w(_w), h(_h), c(_c) {
    tab = new double[c * h * 2];
  }

  ~Polyphone() {
    delete[] tab;
  }

  void from_au(const int8_t *au);
  void to_au(int8_t *au);

  void from_rgb(const uint8_t *rgb);
  void to_rgb(uint8_t *rgb);

  void mask(double level);
};

}

#endif

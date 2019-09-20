#ifndef __MAKEMORE_SOUNDPIC_HH__
#define __MAKEMORE_SOUNDPIC_HH__ 1

#include <stdint.h>

namespace makemore {

struct Soundpic {
  unsigned int w, h;
  double *tab;

  Soundpic(unsigned int _w, unsigned int _h) : w(_w), h(_h) {
    tab = new double[w * h];
  }

  ~Soundpic() {
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

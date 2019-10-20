#ifndef __MAKEMORE_MORK_HH__
#define __MAKEMORE_MORK_HH__ 1

#include <stdint.h>

#include <string>

namespace makemore {

extern const uint8_t mork_rgb[32 * 48 * 3];

struct Mork {
  unsigned int scale;
  uint8_t *rgb;

  Mork(const std::string &_pngfn, unsigned int _scale);
  ~Mork();

  void print(const std::string &x, class Partrait *prt, bool chop = true);
};

}

#endif

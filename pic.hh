#ifndef __MAKEMORE_PIC_HH__
#define __MAKEMORE_PIC_HH__ 1

#include <stdlib.h>
#include <string.h>

#include <vector>
#include <string>
#include <set>

#include "point.hh"
#include "triangle.hh"
#include "pose.hh"
#include "hashbag.hh"

namespace makemore {

struct Pic {
  unsigned int w, h;
  uint8_t *rgb;

  Pic() {
    w = h = 0;
    rgb = NULL;
  }

  Pic(unsigned int _w, unsigned int _h) {
    w = _w;
    h = _h;
    if (w && h) {
      rgb = new uint8_t[w * h * 3];
    } else {
      rgb = NULL;
    }
  }

  Pic(const Pic &pic) {
    w = pic.w;
    h = pic.h;

    if (pic.rgb) {
      assert(w > 0 && h > 0);
      rgb = new uint8_t[w * h * 3];
      memcpy(rgb, pic.rgb, w * h * 3);
    } else {
      rgb = NULL;
      assert(w == 0 && h == 0);
    }
  }

  void clear() {
    if (rgb)
      delete[] rgb;
    rgb = NULL;
    w = h = 0;
  }

  ~Pic() {
    if (rgb)
      delete[] rgb;
  }

  void load(const std::string &fn);
  void load(FILE *fp);
  void save(const std::string &fn) const;
  void save(FILE *) const;

  void to_png(std::string *png) const;
  void from_png(const std::string &png);

  void reflect();

  void paste(const Pic &pic, int x, int y);
};

}

#endif

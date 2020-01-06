#define __MAKEMORE_PIC_CC__ 1

#include <stdlib.h>
#include <string.h>

#include <string>

#include "pic.hh"
#include "strutils.hh"
#include "imgutils.hh"
#include "triangle.hh"
#include "cudamem.hh"
#include "numutils.hh"
#include "strutils.hh"

namespace makemore {

void Pic::load(const std::string &fn) {
  clear();

  std::string png = makemore::slurp(fn);
  from_png(png);
}

void Pic::load(FILE *fp) {
  clear();

  std::string png = makemore::slurp(fp);
  from_png(png);
}

void Pic::from_png(const std::string &png) {
  bool ret = pngrgb(png, &w, &h, &rgb);
  assert(ret);
}

void Pic::save(const std::string &fn) const {
  std::string png;
  to_png(&png);
  makemore::spit(png, fn);
}

void Pic::save(FILE *fp) const {
  std::string png;
  to_png(&png);
  makemore::spit(png, fp);
}
  

void Pic::to_png(std::string *png) const {
  bool ret = rgbpng(rgb, w, h, png);
  assert(ret);
}

void Pic::reflect() {
  unsigned int w2 = w / 2;
  for (unsigned int y = 0; y < h; ++y) {
    for (unsigned int x = 0; x < w2; ++x) { 
      for (unsigned int c = 0; c < 3; ++c) {
        std::swap(rgb[y * w * 3 + x * 3 + c], rgb[y * w * 3 + (w - 1 - x) * 3 + c]);
      }
    }
  }
}

void Pic::paste(const Pic &pic, int x0, int y0) {
  int x1 = x0 + pic.w;
  int y1 = y0 + pic.h;

  for (int y = y0; y < y1; ++y) {
    if (y < 0 || y >= h)
      continue;
    int py = y - y0;

    for (int x = x0; x < x1; ++x) {
      if (x < 0 || x >= w)
        continue;
      int px = x - x0;

      for (unsigned int c = 0; c < 3; ++c) {
        rgb[y * w * 3 + x * 3 + c] = pic.rgb[py * pic.w * 3 + px * 3 + c];
      }
    }
  }
}
     
}

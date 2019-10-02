#ifndef __PPM_HH__
#define __PPM_HH__ 1

#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include <vector>

namespace makemore {

struct PPM {
  uint32_t w, h;
  uint8_t *data;

  PPM() {
    w = h = 0;
    data = NULL;
  }

  ~PPM() {
    if (data)
      delete[] data;
  }

  PPM(unsigned int _w, unsigned int _h, uint8_t v = 0) {
    w = h = 0;
    data = NULL;
    make(_w, _h, v);
  }

  PPM(const PPM &ppm) {
    w = ppm.w;
    h = ppm.h;
    data = new uint8_t[w * h * 3];
    memcpy(data, ppm.data, w * h * 3);
  }

  bool read(FILE *fp);
  void write(FILE *fp);

  void vectorize(std::vector<double> *);
  void vectorize(std::vector<uint8_t> *);
  void vectorize(uint8_t *);
  void unvectorize(const uint8_t *, unsigned int, unsigned int);
  void unvectorize(const std::vector<double> &, unsigned int, unsigned int);
  void unvectorize(const double *, unsigned int, unsigned int);
  void unvectorizegray(const double *, unsigned int, unsigned int);

  void shrink();
  void blurge();
  void zoom();
  void rawzoom();
  void pad();

  void border(unsigned int x0, unsigned int y0, unsigned int x1, unsigned int y1, uint8_t cr = 0, uint8_t cg = 0, uint8_t cb = 255);

  void paste(const uint8_t *vec, unsigned int vw, unsigned int vh, unsigned int x0, unsigned int y0);
  void pastealpha(const uint8_t *vec, const uint8_t *a, unsigned int vw, unsigned int vh, unsigned int x0, unsigned int y0);

  void pastelab(const double *vec, unsigned int vw, unsigned int vh, unsigned int x0, unsigned int y0);
  void pastelab(const uint8_t *vec, unsigned int vw, unsigned int vh, unsigned int x0, unsigned int y0);
  void cutlab(double *vec, unsigned int vw, unsigned int vh, unsigned int x0, unsigned int y0);
  void make(unsigned int _w, unsigned int _h, uint8_t v);

  void write_jpeg(FILE *);
  bool read_jpeg(const std::string &);
  void write_jpeg(std::string *);

  double centerlight();
};

}

#endif

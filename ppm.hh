#ifndef __PPM_HH__
#define __PPM_HH__ 1

#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include <vector>

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

  PPM(const PPM &ppm) {
    w = ppm.w;
    h = ppm.h;
    data = new uint8_t[w * h * 3];
    memcpy(data, ppm.data, w * h * 3);
  }

  bool read(FILE *fp);
  void write(FILE *fp);

  void vectorize(std::vector<double> *);
  void unvectorize(const std::vector<double> &, unsigned int, unsigned int);
  void unvectorize(const double *, unsigned int, unsigned int);
  void unvectorizegray(const double *, unsigned int, unsigned int);

  void shrink();
  void blurge();
  void zoom();
  void rawzoom();
  void pad();
};

void rgbtoxyz(uint8_t r, uint8_t g, uint8_t b, double *xp, double *yp, double *zp);
void xyztolab(double x, double y, double z, double *lp, double *ap, double *bp);
void xyztorgb(double x, double y, double z, uint8_t *r, uint8_t *g, uint8_t *b);
void labtoxyz(double l, double a, double b, double *xp, double *yp, double *zp);

inline void rgbtolab(uint8_t r, uint8_t g, uint8_t b, double *lp, double *ap, double *bp) {
  double x, y, z;
  rgbtoxyz(r, g, b, &x, &y, &z);
  xyztolab(x, y, z, lp, ap, bp);
}

inline void labtorgb(double l, double a, double b, uint8_t *rp, uint8_t *gp, uint8_t *bp) {
  double x, y, z;
  labtoxyz(l, a, b, &x, &y, &z);
  xyztorgb(x, y, z, rp, gp, bp);
}

#endif

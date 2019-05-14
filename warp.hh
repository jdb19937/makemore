#ifndef __MAKEMORE_WARP_HH__
#define __MAKEMORE_WARP_HH__ 1

#include <stdint.h>

namespace makemore {

extern void warp(double *src, double wx, double wy, double wz, double wr, int w, int h, double *dst);

extern void iwarp(double *src, int dx, int dy, int w, int h, double *dst);

extern void jwarp(double *src,
  int w, int h,
  int x0, int y0, int x1, int y1,
  int dw, int dh,
  double *dst
);

extern void kwarp(double *src,
  int w, int h,
  int x0, int y0, int x1, int y1, int x2, int y2,
  int *px0, int *py0, int *px1, int *py1, int *px2, int *py2,
  int dw, int dh,
  double *dst
);

extern void kwarp(const uint8_t *src,
  int w, int h,
  int x0, int y0, int x1, int y1, int x2, int y2,
  int *px0, int *py0, int *px1, int *py1, int *px2, int *py2,
  int dw, int dh,
  uint8_t *dst
);

}

#endif

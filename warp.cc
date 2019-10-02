#define __MAKEMORE_WARP_CC 1
#include "warp.hh"

#include <assert.h>
#include <string.h>
#include <math.h>

namespace makemore {

void warp(double *src, double wx, double wy, double wz, double wr, int w, int h, double *dst) {
  double dx = ((wx - 0.5) * 16.0);
  double dy = ((wy - 0.5) * 16.0);
  double z = 1.0 + (wz - 0.5) * 0.1;
  double th = (wr - 0.5) * 0.25;
  double costh = cos(th);
  double sinth = sin(th);

  double cx = ((double)w) / 2.0;
  double cy = ((double)h) / 2.0;

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {

      double qx = x - cx;
      double qy = y - cy;

      qx *= z;
      qy *= z;

      double rx = dx + cx + costh * qx - sinth * qy;
      double ry = dy + cy + sinth * qx + costh * qy;

      int rx0 = floorl(rx), rx1 = rx0 + 1;
      int ry0 = floorl(ry), ry1 = ry0 + 1;
      double bx = rx - rx0;
      double by = ry - ry0;
      if (rx0 < 0) { rx0 = 0; } if (rx0 >= w) { rx0 = w - 1; }
      if (rx1 < 0) { rx1 = 0; } if (rx1 >= w) { rx1 = w - 1; }
      if (ry0 < 0) { ry0 = 0; } if (ry0 >= h) { ry0 = h - 1; }
      if (ry1 < 0) { ry1 = 0; } if (ry1 >= h) { ry1 = h - 1; }

      for (int c = 0; c < 3; ++c) {
        *dst++ = 
          (1.0-bx) * (1.0-by) * src[ry0 * w * 3 + rx0 * 3 + c] +
          (bx) * (1.0-by) * src[ry0 * w * 3 + rx1 * 3 + c] +
          (1.0-bx) * (by) * src[ry1 * w * 3 + rx0 * 3 + c] +
          (bx) * (by) * src[ry1 * w * 3 + rx1 * 3 + c];
      }
    }
  }
}


void iwarp(double *src, int dx, int dy, int w, int h, double *dst) {
  if (dx == 0 && dy == 0) {
    memcpy(dst, src, w * h * 3 * sizeof(double));
    return;
  }

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {

      int rx = x + dx;
      int ry = y + dy;
      if (rx < 0) { rx = 0; } if (rx >= w) { rx = w - 1; }
      if (ry < 0) { ry = 0; } if (ry >= h) { ry = h - 1; }

      for (int c = 0; c < 3; ++c) {
        *dst++ = src[ry * w * 3 + rx * 3 + c];
      }
    }
  }
}


void jwarp(double *src,
  int w, int h,
  int x0, int y0, int x1, int y1,
  int dw, int dh,
  double *dst
) {

  int x2 = x0 - (y1 - y0);
  int y2 = y0 + (x1 - x0);

  for (int y = 0; y < dh; ++y) {
    for (int x = 0; x < dw; ++x) {
      double rx = (double)x0 + (double)(x1 - x0) * (double)x / (double)dw + (double)(x2 - x0) * (double)y / (double)dh;
      double ry = (double)y0 + (double)(y1 - y0) * (double)x / (double)dw + (double)(y2 - y0) * (double)y / (double)dh;

      int rx0 = floorl(rx), rx1 = rx0 + 1;
      int ry0 = floorl(ry), ry1 = ry0 + 1;
      double bx = rx - rx0;
      double by = ry - ry0;
      if (rx0 < 0) { rx0 = 0; } if (rx0 >= w) { rx0 = w - 1; }
      if (rx1 < 0) { rx1 = 0; } if (rx1 >= w) { rx1 = w - 1; }
      if (ry0 < 0) { ry0 = 0; } if (ry0 >= h) { ry0 = h - 1; }
      if (ry1 < 0) { ry1 = 0; } if (ry1 >= h) { ry1 = h - 1; }

      for (int c = 0; c < 3; ++c) {
        *dst++ = 
          (1.0-bx) * (1.0-by) * src[ry0 * w * 3 + rx0 * 3 + c] +
          (bx) * (1.0-by) * src[ry0 * w * 3 + rx1 * 3 + c] +
          (1.0-bx) * (by) * src[ry1 * w * 3 + rx0 * 3 + c] +
          (bx) * (by) * src[ry1 * w * 3 + rx1 * 3 + c];
      }
    }
  }
}

void kwarp(double *src,
  int w, int h,
  int x0, int y0, int x1, int y1, int x2, int y2,
  int *px0, int *py0, int *px1, int *py1, int *px2, int *py2,
  int dw, int dh,
  double *dst
) {
  if (px0) {
    assert(px0 && py0 && px1 && py1 && px2 && py2);

    // CForm[{x,y} /. Solve[{
    //   rx == x0 + (x1 - x0) * x / dw + (x2 - x0) * y / dh,
    //   ry == y0 + (y1 - y0) * x / dw + (y2 - y0) * y / dh
    // }, {x, y}][[1]]]


    double rx = 0, ry = 0;
    *px0 = -((dw*(ry*x0 - ry*x2 - rx*y0 + x2*y0 + rx*y2 - x0*y2))/(x1*y0 - x2*y0 - x0*y1 + x2*y1 + x0*y2 - x1*y2));
    *py0 = -((-(dh*ry*x0) + dh*ry*x1 + dh*rx*y0 - dh*x1*y0 - dh*rx*y1 + dh*x0*y1)/(x1*y0 - x2*y0 - x0*y1 + x2*y1 + x0*y2 - x1*y2));

    rx = w; ry = 0;
    *px1 = -((dw*(ry*x0 - ry*x2 - rx*y0 + x2*y0 + rx*y2 - x0*y2))/(x1*y0 - x2*y0 - x0*y1 + x2*y1 + x0*y2 - x1*y2));
    *py1 = -((-(dh*ry*x0) + dh*ry*x1 + dh*rx*y0 - dh*x1*y0 - dh*rx*y1 + dh*x0*y1)/(x1*y0 - x2*y0 - x0*y1 + x2*y1 + x0*y2 - x1*y2));

    rx = 0; ry = h;
    *px2 = -((dw*(ry*x0 - ry*x2 - rx*y0 + x2*y0 + rx*y2 - x0*y2))/(x1*y0 - x2*y0 - x0*y1 + x2*y1 + x0*y2 - x1*y2));
    *py2 = -((-(dh*ry*x0) + dh*ry*x1 + dh*rx*y0 - dh*x1*y0 - dh*rx*y1 + dh*x0*y1)/(x1*y0 - x2*y0 - x0*y1 + x2*y1 + x0*y2 - x1*y2));

  }

  for (int y = 0; y < dh; ++y) {
    for (int x = 0; x < dw; ++x) {
      double rx = (double)x0 + (double)(x1 - x0) * (double)x / (double)dw + (double)(x2 - x0) * (double)y / (double)dh;
      double ry = (double)y0 + (double)(y1 - y0) * (double)x / (double)dw + (double)(y2 - y0) * (double)y / (double)dh;

      int rx0 = floorl(rx), rx1 = rx0 + 1;
      int ry0 = floorl(ry), ry1 = ry0 + 1;
      double bx = rx - rx0;
      double by = ry - ry0;
      if (rx0 < 0) { rx0 = 0; } if (rx0 >= w) { rx0 = w - 1; }
      if (rx1 < 0) { rx1 = 0; } if (rx1 >= w) { rx1 = w - 1; }
      if (ry0 < 0) { ry0 = 0; } if (ry0 >= h) { ry0 = h - 1; }
      if (ry1 < 0) { ry1 = 0; } if (ry1 >= h) { ry1 = h - 1; }

      for (int c = 0; c < 3; ++c) {
        *dst++ = 
          (1.0-bx) * (1.0-by) * src[ry0 * w * 3 + rx0 * 3 + c] +
          (bx) * (1.0-by) * src[ry0 * w * 3 + rx1 * 3 + c] +
          (1.0-bx) * (by) * src[ry1 * w * 3 + rx0 * 3 + c] +
          (bx) * (by) * src[ry1 * w * 3 + rx1 * 3 + c];
      }
    }
  }
}

void kwarp(const uint8_t *src,
  int w, int h,
  int x0, int y0, int x1, int y1, int x2, int y2,
  int *px0, int *py0, int *px1, int *py1, int *px2, int *py2,
  int dw, int dh,
  uint8_t *dst,

  const uint8_t *alphasrc,
  uint8_t *alphadst
) {
  if (px0) {
    assert(px0 && py0 && px1 && py1 && px2 && py2);

    // CForm[{x,y} /. Solve[{
    //   rx == x0 + (x1 - x0) * x / dw + (x2 - x0) * y / dh,
    //   ry == y0 + (y1 - y0) * x / dw + (y2 - y0) * y / dh
    // }, {x, y}][[1]]]


    double rx = 0, ry = 0;
    *px0 = -((dw*(ry*x0 - ry*x2 - rx*y0 + x2*y0 + rx*y2 - x0*y2))/(x1*y0 - x2*y0 - x0*y1 + x2*y1 + x0*y2 - x1*y2));
    *py0 = -((-(dh*ry*x0) + dh*ry*x1 + dh*rx*y0 - dh*x1*y0 - dh*rx*y1 + dh*x0*y1)/(x1*y0 - x2*y0 - x0*y1 + x2*y1 + x0*y2 - x1*y2));

    rx = w; ry = 0;
    *px1 = -((dw*(ry*x0 - ry*x2 - rx*y0 + x2*y0 + rx*y2 - x0*y2))/(x1*y0 - x2*y0 - x0*y1 + x2*y1 + x0*y2 - x1*y2));
    *py1 = -((-(dh*ry*x0) + dh*ry*x1 + dh*rx*y0 - dh*x1*y0 - dh*rx*y1 + dh*x0*y1)/(x1*y0 - x2*y0 - x0*y1 + x2*y1 + x0*y2 - x1*y2));

    rx = 0; ry = h;
    *px2 = -((dw*(ry*x0 - ry*x2 - rx*y0 + x2*y0 + rx*y2 - x0*y2))/(x1*y0 - x2*y0 - x0*y1 + x2*y1 + x0*y2 - x1*y2));
    *py2 = -((-(dh*ry*x0) + dh*ry*x1 + dh*rx*y0 - dh*x1*y0 - dh*rx*y1 + dh*x0*y1)/(x1*y0 - x2*y0 - x0*y1 + x2*y1 + x0*y2 - x1*y2));

  }

  for (int y = 0; y < dh; ++y) {
    for (int x = 0; x < dw; ++x) {
      double rx = (double)x0 + (double)(x1 - x0) * (double)x / (double)dw + (double)(x2 - x0) * (double)y / (double)dh;
      double ry = (double)y0 + (double)(y1 - y0) * (double)x / (double)dw + (double)(y2 - y0) * (double)y / (double)dh;

      int rx0 = floorl(rx), rx1 = rx0 + 1;
      int ry0 = floorl(ry), ry1 = ry0 + 1;
      double bx = rx - rx0;
      double by = ry - ry0;
      if (rx0 < 0) { rx0 = 0; } if (rx0 >= w) { rx0 = w - 1; }
      if (rx1 < 0) { rx1 = 0; } if (rx1 >= w) { rx1 = w - 1; }
      if (ry0 < 0) { ry0 = 0; } if (ry0 >= h) { ry0 = h - 1; }
      if (ry1 < 0) { ry1 = 0; } if (ry1 >= h) { ry1 = h - 1; }

      for (int c = 0; c < 3; ++c) {
        *dst++ =
          (1.0-bx) * (1.0-by) * (double)src[ry0 * w * 3 + rx0 * 3 + c] +
          (bx) * (1.0-by) * (double)src[ry0 * w * 3 + rx1 * 3 + c] +
          (1.0-bx) * (by) * (double)src[ry1 * w * 3 + rx0 * 3 + c] +
          (bx) * (by) * (double)src[ry1 * w * 3 + rx1 * 3 + c];
      }

      if (alphadst && alphasrc) {
        *alphadst++ =
          (1.0-bx) * (1.0-by) * (double)alphasrc[ry0 * w + rx0] +
          (bx) * (1.0-by) * (double)alphasrc[ry0 * w + rx1] +
          (1.0-bx) * (by) * (double)alphasrc[ry1 * w + rx0] +
          (bx) * (by) * (double)alphasrc[ry1 * w + rx1];
      }
    }
  }
}

void kwarpover(const uint8_t *src,
  int w, int h,
  int x0, int y0, int x1, int y1, int x2, int y2,
  int *px0, int *py0, int *px1, int *py1, int *px2, int *py2,
  int dw, int dh,
  uint8_t *dst,
  const uint8_t *alphasrc
) {
  if (px0) {
    assert(px0 && py0 && px1 && py1 && px2 && py2);

    // CForm[{x,y} /. Solve[{
    //   rx == x0 + (x1 - x0) * x / dw + (x2 - x0) * y / dh,
    //   ry == y0 + (y1 - y0) * x / dw + (y2 - y0) * y / dh
    // }, {x, y}][[1]]]


    double rx = 0, ry = 0;
    *px0 = -((dw*(ry*x0 - ry*x2 - rx*y0 + x2*y0 + rx*y2 - x0*y2))/(x1*y0 - x2*y0 - x0*y1 + x2*y1 + x0*y2 - x1*y2));
    *py0 = -((-(dh*ry*x0) + dh*ry*x1 + dh*rx*y0 - dh*x1*y0 - dh*rx*y1 + dh*x0*y1)/(x1*y0 - x2*y0 - x0*y1 + x2*y1 + x0*y2 - x1*y2));

    rx = w; ry = 0;
    *px1 = -((dw*(ry*x0 - ry*x2 - rx*y0 + x2*y0 + rx*y2 - x0*y2))/(x1*y0 - x2*y0 - x0*y1 + x2*y1 + x0*y2 - x1*y2));
    *py1 = -((-(dh*ry*x0) + dh*ry*x1 + dh*rx*y0 - dh*x1*y0 - dh*rx*y1 + dh*x0*y1)/(x1*y0 - x2*y0 - x0*y1 + x2*y1 + x0*y2 - x1*y2));

    rx = 0; ry = h;
    *px2 = -((dw*(ry*x0 - ry*x2 - rx*y0 + x2*y0 + rx*y2 - x0*y2))/(x1*y0 - x2*y0 - x0*y1 + x2*y1 + x0*y2 - x1*y2));
    *py2 = -((-(dh*ry*x0) + dh*ry*x1 + dh*rx*y0 - dh*x1*y0 - dh*rx*y1 + dh*x0*y1)/(x1*y0 - x2*y0 - x0*y1 + x2*y1 + x0*y2 - x1*y2));

  }

  for (int y = 0; y < dh; ++y) {
    for (int x = 0; x < dw; ++x) {
      double rx = (double)x0 + (double)(x1 - x0) * (double)x / (double)dw + (double)(x2 - x0) * (double)y / (double)dh;
      double ry = (double)y0 + (double)(y1 - y0) * (double)x / (double)dw + (double)(y2 - y0) * (double)y / (double)dh;

      int rx0 = floorl(rx), rx1 = rx0 + 1;
      int ry0 = floorl(ry), ry1 = ry0 + 1;
      double bx = rx - rx0;
      double by = ry - ry0;
      bool ab = 0;
      if (rx0 < 0) { ab = 1; } if (rx0 >= w) { ab = 1; }
      if (rx1 < 0) { ab = 1; } if (rx1 >= w) { ab = 1; }
      if (ry0 < 0) { ab = 1; } if (ry0 >= h) { ab = 1; }
      if (ry1 < 0) { ab = 1; } if (ry1 >= h) { ab = 1; }
      if (ab) { dst += 3; continue; }

      
      double a = 1.0;
      if (alphasrc)
        a = alphasrc[ry0 * w + rx0] / 256.0;
//a = (a - 0.5) / 0.4;
//a = (a - 0.5) / 0.01;
a = (a - 0.2) / 0.6;
if (a > 1.0) a = 1.0; 
if (a < 0.0) a = 0.0;

      for (int c = 0; c < 3; ++c) {
        double q =  
          (1.0-bx) * (1.0-by) * (double)src[ry0 * w * 3 + rx0 * 3 + c] +
          (bx) * (1.0-by) * (double)src[ry0 * w * 3 + rx1 * 3 + c] +
          (1.0-bx) * (by) * (double)src[ry1 * w * 3 + rx0 * 3 + c] +
          (bx) * (by) * (double)src[ry1 * w * 3 + rx1 * 3 + c];

        dst[0] = q * a + (1 - a) * dst[0];
        ++dst;
      }
    }
  }
}


}

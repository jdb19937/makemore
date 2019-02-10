#define __MAKEMORE_WARP_CC 1
#include "warp.hh"

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

}

#define __MAKEMORE_JULIA_CU__ 1
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "cudamem.hh"
#include "fractals.hh"

namespace makemore {

struct xent {
  double q;
  unsigned int i;
};

static int xcmp(const void *av, const void *bv) {
  const struct xent *a = (const struct xent *)av;
  const struct xent *b = (const struct xent *)bv;
  if (b->q > a->q) {
    return -1;
  } else if (b->q < a->q) {
    return 1;
  } else {
    return 0;
  }
};


void fracrgb(const double *buf, uint8_t *rgb) {
  struct xent *xbuf = new struct xent[65536];
  for (unsigned int i = 0; i < 65536; ++i) {
    xbuf[i].i = i;
    xbuf[i].q = buf[i];
  }
  qsort(xbuf, 65536, sizeof(struct xent), xcmp);

  for (unsigned int i = 0; i < 65536; ++i) {
    int v = i / 256;
    int j = xbuf[i].i;
    
    rgb[3 * j + 0] = v;
    rgb[3 * j + 1] = v;
    rgb[3 * j + 2] = v;
  }

  delete[] xbuf;
}


__global__ void gpu_julia(double *buf, double ca, double cb) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 0 || i >= 65536)
    return;

  int x = i % 256;
  int y = i / 256;

  double za, zb;
  za = 1 * (((double)x / 128.0) - 1.0);
  zb = 1 * (((double)y / 128.0) - 1.0);

  unsigned int m = 512;
  unsigned int n = 0;
  double d2 = za * za + zb * zb;
  double wa, wb;

  while (d2 < 4.0 && n < m) {
    wa = za * za - zb * zb;
    wb = 2.0 * zb * za;
    wa += ca;
    wb += cb;
    za = wa;
    zb = wb;

    d2 = za * za + zb * zb;
    ++n;
  }

  double q = (double)n;
  if (n < m) {
    q += 1.0 - log(log(d2) / log(4.0));
  }
  q *= 255.0 / m;

  buf[i] = q;
}

void julia(uint8_t *rgb, double ca, double cb) {
  double *cubuf;
  cumake(&cubuf, 65536);

  int bs = 256;
  int gs = 256;
  gpu_julia<<<gs, bs>>>(cubuf, ca, cb);

  double *buf = new double[65536];
  decude(cubuf, 65536, buf);
  cufree(cubuf);

  fracrgb(buf, rgb);

  delete[] buf;
}

__global__ void gpu_burnship(double *buf) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 0 || i >= 65536)
    return;

  int x = i % 256;
  int y = i / 256;

  double za = 0, zb = 0;

  double ca, cb;
  ca = -0.5 + 1 * (((double)x / 128.0) - 1.0);
  cb = -0.5 + 1 * (((double)y / 128.0) - 1.0);

  unsigned int m = 255;
  unsigned int n = 0;
  double d2 = za * za + zb * zb;
  double wa, wb;

  while (d2 < 4.0 && n < m) {
    if (za < 0)
      za = -za;
    if (zb < 0)
      zb = -zb;
    wa = za * za - zb * zb;
    wb = 2.0 * zb * za;
    wa += ca;
    wb += cb;
    za = wa;
    zb = wb;

    d2 = za * za + zb * zb;
    ++n;
  }

  double q = (double)n;
  if (n < m) {
    q += 1.0 - log(log(d2) / log(4.0));
  }

  buf[i] = q;
}

void burnship(uint8_t *rgb) {
  double *cubuf;
  cumake(&cubuf, 65536);

  int bs = 256;
  int gs = 256;
  gpu_burnship<<<gs, bs>>>(cubuf);

  double *buf = new double[65536];
  decude(cubuf, 65536, buf);
  cufree(cubuf);

  fracrgb(buf, rgb);

  delete[] buf;
}

__global__ void gpu_mandelbrot(double *buf) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 0 || i >= 65536)
    return;

  int x = i % 256;
  int y = i / 256;

  double za = 0, zb = 0;

  double ca, cb;
  ca = -0.5 + 1 * (((double)y / 128.0) - 1.0);
  cb = 1 * (((double)x / 128.0) - 1.0);

  unsigned int m = 255;
  unsigned int n = 0;
  double d2 = za * za + zb * zb;
  double wa, wb;

  while (d2 < 4.0 && n < m) {
    wa = za * za - zb * zb;
    wb = 2.0 * zb * za;
    wa += ca;
    wb += cb;
    za = wa;
    zb = wb;

    d2 = za * za + zb * zb;
    ++n;
  }

  double q = (double)n;
  if (n < m) {
    q += 1.0 - log(log(d2) / log(4.0));
  }

  buf[i] = q;
}

void mandelbrot(uint8_t *rgb) {
  double *cubuf;
  cumake(&cubuf, 65536);

  int bs = 256;
  int gs = 256;
  gpu_mandelbrot<<<gs, bs>>>(cubuf);

  double *buf = new double[65536];
  decude(cubuf, 65536, buf);
  cufree(cubuf);

  fracrgb(buf, rgb);

  delete[] buf;
}


};



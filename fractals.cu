#define __MAKEMORE_JULIA_CU__ 1
#include <math.h>
#include <stdio.h>
#include <stdint.h>

#include "cudamem.hh"
#include "fractals.hh"

namespace makemore {

__global__ void gpu_julia(double *buf, double ca, double cb) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 0 || i >= 65536)
    return;

  int x = i % 256;
  int y = i / 256;

  double za, zb;
  za = 2 * (((double)x / 128.0) - 1.0);
  zb = 2 * (((double)y / 128.0) - 1.0);

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

void julia(uint8_t *rgb, double ca, double cb) {
  double *cubuf;
  cumake(&cubuf, 65536);

  int bs = 256;
  int gs = 256;
  gpu_julia<<<gs, bs>>>(cubuf, ca, cb);

  double *buf = new double[65536];
  decude(cubuf, 65536, buf);
  cufree(cubuf);

  double m0 = -1;
  double m1 = -1;
  for (unsigned int i = 0; i < 65536; ++i) {
    if (m0 < 0 || buf[i] < m0)
      m0 = buf[i];
    if (m1 < 0 || buf[i] > m1)
      m1 = buf[i];
  }

  for (unsigned int i = 0; i < 65536; ++i) {
    rgb[3 * i + 0] = (uint8_t)(255.0 * (buf[i] - m0) / (m1 - m0));
    rgb[3 * i + 1] = (uint8_t)(255.0 * (buf[i] - m0) / (m1 - m0));
    rgb[3 * i + 2] = (uint8_t)(255.0 * (buf[i] - m0) / (m1 - m0));
  }

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
  ca = 2 * (((double)x / 128.0) - 1.0);
  cb = 2 * (((double)y / 128.0) - 1.0);

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

  double m0 = -1;
  double m1 = -1;
  for (unsigned int i = 0; i < 65536; ++i) {
    if (m0 < 0 || buf[i] < m0)
      m0 = buf[i];
    if (m1 < 0 || buf[i] > m1)
      m1 = buf[i];
  }

  for (unsigned int i = 0; i < 65536; ++i) {
    rgb[3 * i + 0] = (uint8_t)(255.0 * (buf[i] - m0) / (m1 - m0));
    rgb[3 * i + 1] = (uint8_t)(255.0 * (buf[i] - m0) / (m1 - m0));
    rgb[3 * i + 2] = (uint8_t)(255.0 * (buf[i] - m0) / (m1 - m0));
  }

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
  ca = 2 * (((double)x / 128.0) - 1.0);
  cb = 2 * (((double)y / 128.0) - 1.0);

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

  double m0 = -1;
  double m1 = -1;
  for (unsigned int i = 0; i < 65536; ++i) {
    if (m0 < 0 || buf[i] < m0)
      m0 = buf[i];
    if (m1 < 0 || buf[i] > m1)
      m1 = buf[i];
  }

  for (unsigned int i = 0; i < 65536; ++i) {
    rgb[3 * i + 0] = (uint8_t)(255.0 * (buf[i] - m0) / (m1 - m0));
    rgb[3 * i + 1] = (uint8_t)(255.0 * (buf[i] - m0) / (m1 - m0));
    rgb[3 * i + 2] = (uint8_t)(255.0 * (buf[i] - m0) / (m1 - m0));
  }

  delete[] buf;
}


};



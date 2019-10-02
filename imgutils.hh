#ifndef __MAKEMORE_IMGUTILS_HH__
#define __MAKEMORE_IMGUTILS_HH__ 1

#include <assert.h>

#include <vector>
#include <string>

namespace makemore {

void rgbtoxyz(uint8_t r, uint8_t g, uint8_t b, double *xp, double *yp, double *zp);
void xyztolab(double x, double y, double z, double *lp, double *ap, double *bp);
void xyztorgb(double x, double y, double z, uint8_t *r, uint8_t *g, uint8_t *b);
void labtoxyz(double l, double a, double b, double *xp, double *yp, double *zp);

inline void rgbtolab(uint8_t r, uint8_t g, uint8_t b, double *lp, double *ap, double *bp) {
  double x, y, z;
  rgbtoxyz(r, g, b, &x, &y, &z);
  xyztolab(x, y, z, lp, ap, bp);
}

inline void rgbtolab(uint8_t r, uint8_t g, uint8_t b, uint8_t *lp, uint8_t *ap, uint8_t *bp) {
  double tmp[3];
  rgbtolab(r, g, b, tmp + 0, tmp + 1, tmp + 2);

  tmp[0] = 255.0 * (tmp[0] / 100.0);
  tmp[1] += 128.0;
  tmp[2] += 128.0;

  *lp = tmp[0] > 255.0 ? 255 : tmp[0] < 0 ? 0 : (uint8_t)(tmp[0]);
  *ap = tmp[1] > 255.0 ? 255 : tmp[1] < 0 ? 0 : (uint8_t)(tmp[1]);
  *bp = tmp[2] > 255.0 ? 255 : tmp[2] < 0 ? 0 : (uint8_t)(tmp[2]);
}

inline void labquant(const double *dlab, unsigned int n, uint8_t *blab) {
  assert(n % 3 == 0);
  for (unsigned int i = 0; i < n; i += 3) {
    double L = dlab[i + 0] * 255.0; if (L < 0) { L = 0; } if (L > 255.0) { L = 255.0; }
    double A = dlab[i + 1] * 100.0 + 128.0; if (A < 0) { A = 0; } if (A > 255.0) { A = 255.0; }
    double B = dlab[i + 2] * 100.0 + 128.0; if (B < 0) { B = 0; } if (B > 255.0) { B = 255.0; }

    blab[i + 0] = (uint8_t)L;
    blab[i + 1] = (uint8_t)A;
    blab[i + 2] = (uint8_t)B;
  }
}

inline void labdequant(const uint8_t *blab, unsigned int n, double *dlab) {
  assert(n % 3 == 0);
  for (unsigned int i = 0; i < n; i += 3) {
    dlab[i + 0] = 0.01 * ((double)blab[i + 0] * 100.0 / 255.0);
    dlab[i + 1] = 0.01 * ((double)blab[i + 1] - 128.0);
    dlab[i + 2] = 0.01 * ((double)blab[i + 2] - 128.0);
  }
}
    

inline void labtorgb(double l, double a, double b, uint8_t *rp, uint8_t *gp, uint8_t *bp) {
  double x, y, z;
  labtoxyz(l, a, b, &x, &y, &z);
  xyztorgb(x, y, z, rp, gp, bp);
}

inline void rgblab(const uint8_t *rgb, unsigned int n, double *lab) {
  assert(n % 3 == 0);
  for (unsigned int i = 0; i < n; i += 3) {
    rgbtolab(rgb[i+0], rgb[i+1], rgb[i+2], &lab[i+0], &lab[i+1], &lab[i+2]);
  }
}

inline void rgbalaba(const uint8_t *rgb, const uint8_t *a, unsigned int n, double *laba) {
  for (unsigned int i = 0; i < n; i++) {
    rgbtolab(rgb[3*i+0], rgb[3*i+1], rgb[3*i+2], &laba[4*i+0], &laba[4*i+1], &laba[4*i+2]);
    laba[4*i+3] = (double)a[i] / 255.0;
  }
}

inline void rgblaba(const uint8_t *rgb, unsigned int n, double *laba) {
  for (unsigned int i = 0; i < n; i++) {
    rgbtolab(rgb[3*i+0], rgb[3*i+1], rgb[3*i+2], &laba[4*i+0], &laba[4*i+1], &laba[4*i+2]);
    laba[4*i+3] = 1.0;
  }
}

inline void labrgb(const double *lab, unsigned int n, uint8_t *rgb) {
  assert(n % 3 == 0);
  for (unsigned int i = 0; i < n; i += 3) {
    labtorgb(lab[i+0], lab[i+1], lab[i+2], &rgb[i+0], &rgb[i+1], &rgb[i+2]);
  }
}

inline void labargba(const double *laba, unsigned int n, uint8_t *rgb, uint8_t *a) {
  for (unsigned int i = 0; i < n; i++) {
    labtorgb(laba[4*i+0], laba[4*i+1], laba[4*i+2], &rgb[3*i+0], &rgb[3*i+1], &rgb[3*i+2]);

    double ax = laba[4*i+3];
    if (ax > 1.0) ax = 1.0;
    if (ax < 0.0) ax = 0.0;
    a[i] = (uint8_t)(ax * 255.0);
  }
}

extern bool imglab(
  const std::string &fmt, 
  const std::string &data,
  unsigned int w,
  unsigned int h,
  uint8_t *lab,
  std::vector<std::string> *tags = NULL
);

bool labimg(
  const uint8_t *lab,
  unsigned int w,
  unsigned int h,
  const std::string &fmt,
  std::string *img,
  const std::vector<std::string> *tags = NULL
);

bool rgbpng(
  const uint8_t *rgb,
  unsigned int w,
  unsigned int h,
  std::string *png,
  const std::vector<std::string> *tags = NULL,
  const uint8_t *alpha = NULL
);
bool labpng(
  const uint8_t *lab,
  unsigned int w,
  unsigned int h,
  std::string *png,
  const std::vector<std::string> *tags = NULL
);

bool pngrgb(
  const std::string &png,
  unsigned int w,
  unsigned int h,
  uint8_t *rgb,
  std::vector<std::string> *tags = NULL
);

bool pngrgb(
  const std::string &png,
  unsigned int *wp,
  unsigned int *hp,
  uint8_t **rgbp,
  std::vector<std::string> *tags = NULL,
  uint8_t **alphap = NULL
);

bool pngrgb(
  const std::string &png,
  unsigned int w,
  unsigned int h,
  double *rgb,
  std::vector<std::string> *tags = NULL
);

bool pnglab(
  const std::string &png,
  unsigned int w,
  unsigned int h,
  uint8_t *lab,
  std::vector<std::string> *tags = NULL
);

void padshift(const uint8_t *in, unsigned int w, unsigned int h, int dx, int dy, uint8_t *out, unsigned int nc = 1);

}

#endif

#ifndef __MAKEMORE_IMGUTILS_HH__
#define __MAKEMORE_IMGUTILS_HH__ 1

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
  *lp = tmp[0] > 255.0 ? 255 : tmp[0] < 0 ? 0 : (uint8_t)(tmp[0] * 256.0);
  *ap = tmp[1] > 255.0 ? 255 : tmp[1] < 0 ? 0 : (uint8_t)(tmp[1] * 256.0);
  *bp = tmp[2] > 255.0 ? 255 : tmp[2] < 0 ? 0 : (uint8_t)(tmp[2] * 256.0);
}

inline void labtorgb(double l, double a, double b, uint8_t *rp, uint8_t *gp, uint8_t *bp) {
  double x, y, z;
  labtoxyz(l, a, b, &x, &y, &z);
  xyztorgb(x, y, z, rp, gp, bp);
}

inline void labtorgb(uint8_t l, uint8_t a, uint8_t b, uint8_t *rp, uint8_t *gp, uint8_t *bp) {
  double x, y, z;
  labtoxyz(
    ((double)l + 0.5) / 256.0,
    ((double)a + 0.5) / 256.0,
    ((double)b + 0.5) / 256.0,
    &x, &y, &z);
  xyztorgb(x, y, z, rp, gp, bp);
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
  std::string *png,
  const std::vector<std::string> *tags = NULL
);

bool labpng(
  const uint8_t *lab,
  unsigned int w,
  unsigned int h,
  std::string *png,
  const std::vector<std::string> *tags = NULL
);

bool pnglab(
  const std::string &png,
  unsigned int w,
  unsigned int h,
  uint8_t *lab,
  std::vector<std::string> *tags = NULL
);


}

#endif

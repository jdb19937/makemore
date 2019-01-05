#include <stdio.h>
#include <ctype.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include <math.h>

#include "ppm.hh"

bool PPM::read(FILE *fp) {
  int ret = getc(fp);
  if (ret == EOF)
    return 0;
  assert(ret == 'P');
  assert(getc(fp) == '6');
  assert(isspace(getc(fp)));

  int c;
  char buf[256], *p = buf;
  do { *p = getc(fp); } while (isspace(*p));
  ++p;

  while (!isspace(*p = getc(fp))) {
    assert(isdigit(*p));
    assert(p - buf < 32);
    ++p;
  }
  *p = 0;
  w = atoi(buf);

  p = buf;
  while (!isspace(*p = getc(fp))) {
    assert(isdigit(*p));
    assert(p - buf < 32);
    ++p;
  }
  *p = 0;
  h = atoi(buf);

  p = buf;
  while (!isspace(*p = getc(fp))) {
    assert(isdigit(*p));
    assert(p - buf < 32);
    ++p;
  }
  *p = 0;
  assert(!strcmp(buf, "255"));

  if (data)
    delete[] data;
  data = new uint8_t[3 * w * h];

  assert(3 * w * h == fread(data, 1, 3 * w * h, fp));
  return 1;
}

void PPM::write(FILE *fp) {
  fprintf(fp, "P6\n%u %u\n255\n", w, h);
  assert(3 * w * h == fwrite(data, 1, 3 * w * h, fp));
  fflush(fp);
}


static double squish(uint8_t x, double d = 0.02) {
  return (d + ((double)x/255.999) * (1 - 2.0 * d));
}

static uint8_t unsquish(double x, double d = 0.02) {
  double y = ((x - d) / (1 - 2.0 * d));
  if (y > 1.0) y = 1.0;
  if (y < 0.0) y = 0.0;
  return (uint8_t)(y * 255.999);
}

void PPM::vectorize(std::vector<double> *vecp) {
  unsigned int n = w * h * 3;
  vecp->resize(n);
  for (unsigned int i = 0; i < n; i += 3) {
    double l, a, b;
    rgbtolab(data[i+0], data[i+1], data[i+2], &l, &a, &b);
    (*vecp)[i+0] = l;
    (*vecp)[i+1] = a;
    (*vecp)[i+2] = b;
  }
}

void PPM::unvectorizegray(const double *vec, unsigned int _w, unsigned int _h) {
  unsigned int n = _w * _h * 3;

  w = _w;
  h = _h;
  if (data)
    delete[] data;
  data = new uint8_t[n];

  for (unsigned int i = 0; i < n; i += 3) {
    uint8_t r, g, b;
    labtorgb(vec[i/3], 0.5, 0.5, &r, &g, &b);
    data[i+0] = r;
    data[i+1] = g;
    data[i+2] = b;
  }
}

void PPM::make(unsigned int _w, unsigned int _h, uint8_t v) {
  unsigned int n = _w * _h * 3;
  w = _w;
  h = _h;

  if (data)
    delete[] data;
  data = new uint8_t[n];

  for (unsigned int i = 0; i < n; ++i)
    data[i] = v;
}

void PPM::pastelab(const double *vec, unsigned int vw, unsigned int vh, unsigned int x0, unsigned int y0) {
  assert(y0 + vh <= h);
  assert(x0 + vw <= w);

  unsigned int i = 0;
  for (unsigned int vy = 0; vy < vh; ++vy) {
    unsigned int y = y0 + vy;
    unsigned int yw3 = y * w * 3;

    for (unsigned int vx = 0; vx < vw; ++vx) {
      unsigned int x = x0 + vx;
      unsigned int x3 = x * 3;

      labtorgb(
        vec[i+0], vec[i+1], vec[i+2], 
        data+yw3+x3+0, data+yw3+x3+1, data+yw3+x3+2
      );

      i += 3;
    }
  }
}

void PPM::unvectorize(const double *vec, unsigned int _w, unsigned int _h) {
  unsigned int n = _w * _h * 3;

  w = _w;
  h = _h;
  if (data)
    delete[] data;
  data = new uint8_t[n];

  for (unsigned int i = 0; i < n; i += 3) {
    uint8_t r, g, b;
    labtorgb(vec[i+0], vec[i+1], vec[i+2], &r, &g, &b);
    data[i+0] = r;
    data[i+1] = g;
    data[i+2] = b;
  }
}

void PPM::unvectorize(const std::vector<double> &vec, unsigned int _w, unsigned int _h) {
  unsigned int n = _w * _h * 3;
  assert(n == vec.size());

  w = _w;
  h = _h;
  if (data)
    delete[] data;
  data = new uint8_t[n];

  for (unsigned int i = 0; i < n; i += 3) {
    uint8_t r, g, b;
    labtorgb(vec[i+0], vec[i+1], vec[i+2], &r, &g, &b);
    data[i+0] = r;
    data[i+1] = g;
    data[i+2] = b;
  }
}

void PPM::pad() {
  unsigned int nw = w + 2;
  unsigned int nh = h + 2;
  uint8_t *ndata = new uint8_t[nw * nh * 3];

  for (unsigned int c = 0; c < 3; ++c) {

  for (unsigned int y = 0; y < nh; ++y) {
    for (unsigned int x = 0; x < nw; ++x) {
      unsigned int zx = x;
      if (zx > 0) zx--;
      if (zx >= w) zx--;

      unsigned int zy = y;
      if (zy > 0) zy--;
      if (zy >= h) zy--;

      ndata[y*3*nw + x*3 + c] = data[zy*3*w + zx*3 + c];
#if 0
      continue;

      if (x > 0 && x < nw - 1 && y > 0 && y < nh - 1) {
        ndata[y*3*nw + x*3 + c] = data[(y-1)*3*w + (x-1)*3 + c];
        continue;
      }
       
      ndata[y*3*nw + x*3 + c] = 0;
#endif
    }
  }

  }

  delete[] data;
  data = ndata;
  w = nw;
  h = nh;
}
        

void PPM::zoom() {
  unsigned int nw = w * 2;
  unsigned int nh = h * 2;

  assert(nw > 0);
  assert(nh > 0);
  assert(data);

  double *vec = new double[nw * nh * 3]();

  for (unsigned int y = 0; y < h; ++y) {
    for (unsigned int x = 0; x < w; ++x) {
      for (unsigned int c = 0; c < 3; ++c) {
        uint8_t z = data[y * w * 3 + x * 3 + c];

        vec[(2*y+0) * nw * 3 + (2*x+0) * 3 + c] = z;

        vec[(2*y+1) * nw * 3 + (2*x+0) * 3 + c] += z/2.0;
        if (y > 0)
          vec[(2*y-1) * nw * 3 + (2*x+0) * 3 + c] += z/2.0;

        vec[(2*y+0) * nw * 3 + (2*x+1) * 3 + c] += z/2.0;
        if (x > 0)
          vec[(2*y+0) * nw * 3 + (2*x-1) * 3 + c] += z/2.0;

        vec[(2*y+1) * nw * 3 + (2*x+1) * 3 + c] += z/4.0;
        if (x > 0)
          vec[(2*y+1) * nw * 3 + (2*x-1) * 3 + c] += z/4.0;
        if (y > 0)
          vec[(2*y-1) * nw * 3 + (2*x+1) * 3 + c] += z/4.0;
        if (x > 0 && y > 0)
          vec[(2*y-1) * nw * 3 + (2*x-1) * 3 + c] += z/4.0;

        if (x == w-1) {
          vec[(2*y+0) * nw * 3 + (2*x+1) * 3 + c] += z/2.0;
          vec[(2*y+1) * nw * 3 + (2*x+1) * 3 + c] += z/4.0;
          if (y > 0)
            vec[(2*y-1) * nw * 3 + (2*x+1) * 3 + c] += z/4.0;
        }

        if (y == h-1) {
          vec[(2*y+1) * nw * 3 + (2*x+0) * 3 + c] += z/2.0;
          vec[(2*y+1) * nw * 3 + (2*x+1) * 3 + c] += z/4.0;
          if (x > 0)
            vec[(2*y+1) * nw * 3 + (2*x-1) * 3 + c] += z/4.0;
        }

        if (x == w-1 && y == h-1) {
          vec[(2*y+1) * nw * 3 + (2*x+1) * 3 + c] += z/4.0;
        }
      }
    }
  }

  uint8_t *ndata = new uint8_t[nw * nh * 3];
  for (int i = nw * nh * 3; i >= 0; --i)
    ndata[i] = vec[i];
  delete[] vec;

  delete[] data;
  data = ndata;
  w = nw;
  h = nh;
}

void PPM::rawzoom() {
  unsigned int nw = w * 2;
  unsigned int nh = h * 2;

  assert(nw > 0);
  assert(nh > 0);
  assert(data);

  double *vec = new double[nw * nh * 3]();

  for (unsigned int y = 0; y < h; ++y) {
    for (unsigned int x = 0; x < w; ++x) {
      for (unsigned int c = 0; c < 3; ++c) {
        double z = data[y * w * 3 + x * 3 + c];

        vec[(2*y+0) * nw * 3 + (2*x+0) * 3 + c] = z;
        vec[(2*y+1) * nw * 3 + (2*x+0) * 3 + c] = z;
        vec[(2*y+0) * nw * 3 + (2*x+1) * 3 + c] = z;
        vec[(2*y+1) * nw * 3 + (2*x+1) * 3 + c] = z;
      }
    }
  }

  uint8_t *ndata = new uint8_t[nw * nh * 3];
  for (int i = nw * nh * 3; i >= 0; --i)
    ndata[i] = vec[i];
  delete[] vec;

  delete[] data;
  data = ndata;
  w = nw;
  h = nh;
}

void PPM::shrink() {
  unsigned int nw = w / 2;
  unsigned int nh = h / 2;

  assert(nw > 0);
  assert(nh > 0);
  assert(data);

  uint8_t *ndata = new uint8_t[nw * nh * 3];

  for (unsigned int y = 0; y < nh; ++y) {
    for (unsigned int x = 0; x < nw; ++x) {
      for (unsigned int c = 0; c < 3; ++c) {

        double z = 0;
        z += data[y * w * 6 + x * 6 + c + 0];
        z += data[y * w * 6 + x * 6 + c + 3];
        z += data[y * w * 6 + x * 6 + c + w * 3 + 0];
        z += data[y * w * 6 + x * 6 + c + w * 3 + 3];
        z /= 4.0;

        ndata[y * nw * 3 + x * 3 + c] = round(z);
      }
    }
  }

  delete[] data;
  data = ndata;
  w = nw;
  h = nh;
}

static inline double pd(double x, double y, double w, double h) {
  double dx = (x - w/2.0) / (w/2.0);
  double dy = (y - h/2.0) / (h/2.0);
  double dx2 = dx * dx;
  double dy2 = dy * dy;
  double d = sqrt(dx2 + dy2);
  d -= 0.5;
  d *= 1.8;
  if (d < 0)
    d = 0;
  if (d > 1)
    d = 1;
  return d;
}

void PPM::blurge() {
  assert(data);

  assert(h % 2 == 0);
  assert(w % 2 == 0);
  uint8_t *ndata = new uint8_t[w * h * 3];

  for (unsigned int y = 0; y < h; y += 2) {
    for (unsigned int x = 0; x < w; x += 2) {
      for (unsigned int c = 0; c < 3; ++c) {
        double z = 0;
        z += data[y * w * 3 + x * 3 + c + 0];
        z += data[y * w * 3 + x * 3 + c + 3];
        z += data[y * w * 3 + x * 3 + c + w * 3 + 0];
        z += data[y * w * 3 + x * 3 + c + w * 3 + 3];
        z /= 4.0;

        double q, d;
        q = data[y * w * 3 + x * 3 + c + 0];
        d = pd(x, y, w, h);
        ndata[y * w * 3 + x * 3 + c + 0] = (1-d)*q + d*z;

        q = data[y * w * 3 + x * 3 + c + 3];
        d = pd(x+1, y, w, h);
        ndata[y * w * 3 + x * 3 + c + 3] = (1-d)*q + d*z;

        q = data[y * w * 3 + x * 3 + c + w * 3 + 0];
        d = pd(x, y+1, w, h);
        ndata[y * w * 3 + x * 3 + c + w * 3 + 0] = (1-d)*q + d*z;

        q = data[y * w * 3 + x * 3 + c + w * 3 + 3];
        d = pd(x+1, y+1, w, h);
        ndata[y * w * 3 + x * 3 + c + w * 3 + 3] = (1-d)*q + d*z;
      }
    }
  }

  delete[] data;
  data = ndata;
}

#ifdef PPMBLURGE_MAIN
int main() {
  PPM p;
  p.read(stdin);
  p.blurge();
  p.write(stdout);
  return 0;
}
#endif


static inline double f(double n) {
   return (n > 0.04045 ? pow((n + 0.055) / 1.055, 2.4) : n / 12.92) * 100.0;
}

void rgbtoxyz(uint8_t r, uint8_t g, uint8_t b, double *xp, double *yp, double *zp) {
  double dr = f((double)r / 255.0);
  double dg = f((double)g / 255.0);
  double db = f((double)b / 255.0);

  *xp = dr * 0.4124 + dg * 0.3576 + db * 0.1805;
  *yp = dr * 0.2126 + dg * 0.7152 + db * 0.0722;
  *zp = dr * 0.0193 + dg * 0.1192 + db * 0.9505;
}

void xyztorgb(double x, double y, double z, uint8_t *rp, uint8_t *gp, uint8_t *bp) {
  double dx = x / 100.0;
  double dy = y / 100.0;
  double dz = z / 100.0;

  double pr = dx * 3.2404542 + dy * -1.5371385 + dz * -0.4985314;
  double pg = dx * -0.9692660 + dy * 1.8760108 + dz * 0.0415560;
  double pb = dx * 0.0556434 + dy * -0.2040259 + dz * 1.0572252;

  double qr = (((pr > 0.0031308) ? (1.055*pow(pr, 1 / 2.4) - 0.055) : (12.92*pr)) * 255.0);
  double qg = (((pg > 0.0031308) ? (1.055*pow(pg, 1 / 2.4) - 0.055) : (12.92*pg)) * 255.0);
  double qb = (((pb > 0.0031308) ? (1.055*pow(pb, 1 / 2.4) - 0.055) : (12.92*pb)) * 255.0);

  if (qr < 0) qr = 0; if (qr > 255.0) qr = 255.0;
  if (qg < 0) qg = 0; if (qg > 255.0) qg = 255.0;
  if (qb < 0) qb = 0; if (qb > 255.0) qb = 255.0;

  *rp = qr;
  *gp = qg;
  *bp = qb;
}

		

static inline double g(double n) {
  const double EPS = 0.008856;
  const double KAPPA = 903.3;
  return n > EPS ? pow(n, 1.0/3.0) : (KAPPA * n + 16.0) / 116.0;
}

//double tl = 0.25;
//double tab = 0.5;

void xyztolab(double x, double y, double z, double *lp, double *ap, double *bp) {
  double dx = g(x / 95.047);
  double dy = g(y / 100.0);
  double dz = g(z / 108.883);

  *lp = 116.0 * dy - 16.0;
  if (*lp < 0)
    *lp = 0;
  *ap = 500.0 * (dx - dy);
  *bp = 200.0 * (dy - dz);

  *lp -= 50.0;
  *lp /= 100.0;
  *ap /= 100.0;
  *bp /= 100.0;

//  *lp *= tl;
//  *ap *= tab;
//  *bp *= tab;

  *lp += 0.5;
  *ap += 0.5;
  *bp += 0.5;
}

void labtoxyz(double l, double a, double b, double *xp, double *yp, double *zp) {
  if (l <  0.0) l = 0.0;
  if (l >= 1.0) l = 1.0;
  l -= 0.5;
//  l /= tl;
  l *= 100.0;
  l += 50.0;

  if (a <  0.0) a = 0.0;
  if (a >= 1.0) a = 1.0;
  a -= 0.5;
//  a /= tab;
  a *= 100.0;

  if (b <  0.0) b = 0.0;
  if (b >= 1.0) b = 1.0;
  b -= 0.5;
//  b /= tab;
  b *= 100.0;
  
  double py = (l + 16.0) / 116.0;
  double px = a / 500.0 + py;
  double pz = py - b / 200.0;

  double x3 = px*px*px;
  double y3 = py*py*py;
  double z3 = pz*pz*pz;

  *xp = ((x3 > 0.008856) ? x3 : ((px - 16.0 / 116.0) / 7.787)) * 95.047;
  *yp = ((y3 > 0.008856) ? y3 : ((py - 16.0 / 116.0) / 7.787)) * 100.0;
  *zp = ((z3 > 0.008856) ? z3 : ((pz - 16.0 / 116.0) / 7.787)) * 108.883;
}


#if PPMTEST_MAIN
int main() {
  PPM p;
  p.read(stdin);
  for (unsigned int y = 0; y < p.h; ++y) {
    for (unsigned int x = 0; x < p.w; ++x) {
      uint8_t r = p.data[y * p.w * 3 + x * 3 + 0];
      uint8_t g =   p.data[y * p.w * 3 + x * 3 + 1];
       uint8_t b =  p.data[y * p.w * 3 + x * 3 + 2];
      double l, a, B;
      rgbtolab(r,g, b,
        &l, &a, &B
      );
      uint8_t r2, g2, b2;
      labtorgb(l, a, B, &r2, &g2, &b2);
      printf("%lf\t%lf\t%lf (%d %d %d)\n", l, a, B, r-r2, g-g2, b-b2);
    }
  }
  return 0;
}
#endif


#if PPMSHRINK_MAIN
int main() {
  PPM p;
  p.read(stdin);
  p.shrink();
  p.write(stdout);
  return 0;
}
#endif

#if PPMZOOM_MAIN
int main() {
  PPM p;
  p.read(stdin);
  p.zoom();
  p.write(stdout);
  return 0;
}
#endif


#if PPMPAD_MAIN
int main(int argc, char **argv) {
  PPM p;
  unsigned int n = 1;
  if (argc > 1)
    n = atoi(argv[1]);

  p.read(stdin);
  for (unsigned int i = 0; i < n; ++i)
    p.pad();
  p.write(stdout);
  return 0;
}
#endif


#if PPMTOLAB_MAIN
int main(int argc, char **argv) {
  PPM p;
  p.read(stdin);
  std::vector<double> v;
  p.vectorize(&v);
  assert(v.size() == p.w * p.h * 3);
  fwrite(v.data(), sizeof(double), v.size(), stdout);
  return 0;
}
#endif

#if PPMTOL_MAIN
int main(int argc, char **argv) {
  PPM p;
  p.read(stdin);
  std::vector<double> v;
  p.vectorize(&v);
  assert(v.size() == p.w * p.h * 3);

  std::vector<double> l;
  l.resize(p.w * p.h);
  for (unsigned int i = 0; i < v.size(); i += 3)
    l[i/3] = v[i];

  fwrite(l.data(), sizeof(double), l.size(), stdout);
  return 0;
}
#endif



#if PPMTOLABTWID_MAIN

typedef std::vector<double> Vec;
void twiddle(const Vec &z, unsigned int w, unsigned int h, Vec *lo, Vec *hi) {
  assert(w % 2 == 0 && h % 2 == 0);
  assert(z.size() == 3 * w * h);

  unsigned int nw = w/2;
  unsigned int nh = h/2;
  lo->resize(nw * nh * 3);
  hi->resize(nw * nh * 9);

  unsigned int ilo = 0, ihi = 0;

  unsigned int w3 = w * 3;
  for (unsigned int y = 0; y < h; y += 2) {
    for (unsigned int x = 0; x < w; x += 2) {
      for (unsigned int c = 0; c < 3; ++c) {
        unsigned int p = y * w3 + x * 3 + c;

        double m = (z[p] + z[p+3] + z[p+w3] + z[p+w3+3]) / 4.0;
        double l = (z[p] + z[p+w3]) / 2.0 - m;
        double t = (z[p] + z[p+3]) / 2.0 - m;
        double s = (z[p] + z[p+w3+3]) / 2.0 - m;

        (*lo)[ilo++] = m;
        (*hi)[ihi++] = 0.5 + l/2.0;
        (*hi)[ihi++] = 0.5 + t/2.0;
        (*hi)[ihi++] = 0.5 + s/2.0;
      }
    }
  }
}

int main(int argc, char **argv) {
  PPM p;
  p.read(stdin);
  std::vector<double> v;
  p.vectorize(&v);
  assert(v.size() == p.w * p.h * 3);
  std::vector<double> lo, hi;
  twiddle(v, p.w, p.h, &lo, &hi);
  fwrite(hi.data(), sizeof(double), hi.size(), stdout);
  fwrite(lo.data(), sizeof(double), lo.size(), stdout);
  return 0;
}
#endif

#if PPMTOLABHI_MAIN
typedef std::vector<double> Vec;
void twiddle(const Vec &z, unsigned int w, unsigned int h, Vec *lo, Vec *hi) {
  assert(w % 2 == 0 && h % 2 == 0);
  assert(z.size() == 3 * w * h);

  unsigned int nw = w/2;
  unsigned int nh = h/2;
  lo->resize(nw * nh * 3);
  hi->resize(nw * nh * 9);

  unsigned int ilo = 0, ihi = 0;

  unsigned int w3 = w * 3;
  for (unsigned int y = 0; y < h; y += 2) {
    for (unsigned int x = 0; x < w; x += 2) {
      for (unsigned int c = 0; c < 3; ++c) {
        unsigned int p = y * w3 + x * 3 + c;

        double m = (z[p] + z[p+3] + z[p+w3] + z[p+w3+3]) / 4.0;
        double l = (z[p] + z[p+w3]) / 2.0 - m;
        double t = (z[p] + z[p+3]) / 2.0 - m;
        double s = (z[p] + z[p+w3+3]) / 2.0 - m;

        (*lo)[ilo++] = m;
        (*hi)[ihi++] = 0.5 + l/2.0;
        (*hi)[ihi++] = 0.5 + t/2.0;
        (*hi)[ihi++] = 0.5 + s/2.0;
      }
    }
  }
}

int main(int argc, char **argv) {
  PPM p;
  p.read(stdin);
  std::vector<double> v;
  p.vectorize(&v);
  assert(v.size() == p.w * p.h * 3);
  std::vector<double> lo, hi;
  twiddle(v, p.w, p.h, &lo, &hi);
  fwrite(hi.data(), sizeof(double), hi.size(), stdout);
  return 0;
}
#endif

#if PPMTOLTWID_MAIN
typedef std::vector<double> Vec;

void twiddle(const Vec &z, unsigned int w, unsigned int h, Vec *lo, Vec *hi) {
  assert(w % 2 == 0 && h % 2 == 0);
  assert(z.size() == 3 * w * h);

  unsigned int nw = w/2;
  unsigned int nh = h/2;
  lo->resize(nw * nh * 3);
  hi->resize(nw * nh * 9);

  unsigned int ilo = 0, ihi = 0;

  unsigned int w3 = w * 3;
  for (unsigned int y = 0; y < h; y += 2) {
    for (unsigned int x = 0; x < w; x += 2) {
      for (unsigned int c = 0; c < 3; ++c) {
        unsigned int p = y * w3 + x * 3 + c;

        double m = (z[p] + z[p+3] + z[p+w3] + z[p+w3+3]) / 4.0;
        double l = (z[p] + z[p+w3]) / 2.0 - m;
        double t = (z[p] + z[p+3]) / 2.0 - m;
        double s = (z[p] + z[p+w3+3]) / 2.0 - m;

        (*lo)[ilo++] = m;
        (*hi)[ihi++] = 0.5 + l/2.0;
        (*hi)[ihi++] = 0.5 + t/2.0;
        (*hi)[ihi++] = 0.5 + s/2.0;
      }
    }
  }
}

int main(int argc, char **argv) {
  PPM p;
  p.read(stdin);
  std::vector<double> v;
  p.vectorize(&v);
  assert(v.size() == p.w * p.h * 3);
  assert(p.w % 2 == 0 && p.h % 2 == 0);

  std::vector<double> lo, hi;
  twiddle(v, p.w, p.h, &lo, &hi);
  assert(hi.size() % 3 == 0);
  assert(lo.size() % 3 == 0);

  std::vector<double> h;
  h.resize(hi.size() / 3);
  for (unsigned int i = 0; i < hi.size(); i += 3)
    h[i/3] = hi[i];

  fwrite(h.data(), sizeof(double), h.size(), stdout);

  std::vector<double> l;
  l.resize(lo.size() / 3);
  for (unsigned int i = 0; i < lo.size(); i += 3)
    l[i/3] = lo[i];

  fwrite(l.data(), sizeof(double), l.size(), stdout);
  return 0;
}
#endif


#if LABTOPPM_MAIN
int main(int argc, char **argv) {
  assert(argc == 3);
  unsigned int w = atoi(argv[1]);
  unsigned int h = atoi(argv[2]);

  std::vector<double> v;
  v.resize(w * h * 3);
  int ret = fread(v.data(), sizeof(double), w * h * 3, stdin);
  assert(ret == w * h * 3);

  PPM p;
  p.unvectorize(v, w, h);
  p.write(stdout);
  return 0;
}
#endif


#if LTOPPM_MAIN
int main(int argc, char **argv) {
  assert(argc == 3);
  unsigned int w = atoi(argv[1]);
  unsigned int h = atoi(argv[2]);

  std::vector<double> v;
  v.resize(w * h);
  int ret = fread(v.data(), sizeof(double), w * h, stdin);
  assert(ret == w * h);

  PPM p;
  p.unvectorizegray(v.data(), w, h);
  p.write(stdout);
  return 0;
}
#endif

#if LABTWIDTOPPM_MAIN
typedef std::vector<double> Vec;

void untwiddle(const Vec &lo, const Vec &hi, unsigned int w, unsigned int h, Vec *z) {
  assert(w % 2 == 0 && h % 2 == 0);
  z->resize(3 * w * h);

  unsigned int nw = w/2;
  unsigned int nh = h/2;
  assert(lo.size() == nw * nh * 3);
  assert(hi.size() == nw * nh * 9);

  unsigned int ilo = 0, ihi = 0;

  unsigned int w3 = w * 3;
  for (unsigned int y = 0; y < h; y += 2) {
    for (unsigned int x = 0; x < w; x += 2) {
      for (unsigned int c = 0; c < 3; ++c) {
        unsigned int p = y * w3 + x * 3 + c;

        double m = lo[ilo++];
        double l = (hi[ihi++] - 0.5) * 2.0;
        double t = (hi[ihi++] - 0.5) * 2.0;
        double s = (hi[ihi++] - 0.5) * 2.0;

        (*z)[p] = m + l + t + s;
        (*z)[p+3] = m - l + t - s;
        (*z)[p+w3] = m + l - t - s;
        (*z)[p+w3+3] = m - l - t + s;
      }
    }
  }
}
int main(int argc, char **argv) {
  assert(argc == 3);
  unsigned int w = atoi(argv[1]);
  unsigned int h = atoi(argv[2]);

  assert(w % 2 == 0 && h % 2 == 0);

  std::vector<double> hi;
  hi.resize((w * h * 3 * 3)/4);
  int ret = fread(hi.data(), sizeof(double), hi.size(), stdin);
  assert(ret == hi.size());

  std::vector<double> lo;
  lo.resize((w * h * 3) / 4);
  ret = fread(lo.data(), sizeof(double), lo.size(), stdin);
  assert(ret == lo.size());

  PPM p;
  std::vector<double> v;
  untwiddle(lo, hi, w, h, &v);
  
  p.unvectorize(v, w, h);
  p.write(stdout);
  return 0;
}
#endif

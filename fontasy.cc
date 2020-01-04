#define __MAKEMORE_FONTASY_CC__ 1
#include "fontasy.hh"
#include "imgutils.hh"
#include "strutils.hh"
#include "superenc.hh"
#include "supergen.hh"
#include "partrait.hh"
#include "parson.hh"
#include "styler.hh"
#include "cholo.hh"
#include "random.hh"
#include "display.hh"
#include "tmutils.hh"
#include "fontnoms.hh"
#include "pic.hh"

#include <SDL2/SDL.h>

namespace makemore {

Phont::Phont() {
  rgb = NULL;
  w = h = 0;
  dim = 0;
}

Phont::Phont(const std::string &fn) {
  rgb = NULL;
  w = h = 0;
  dim = 0;
  load(fn);
}

void Phont::clear() {
  if (rgb) {
    delete[] rgb;
    rgb = NULL;
  }
  w = h = 0;
  dim = 0;
}

void Phont::to_png(std::string *png) const {
  bool ret = rgbpng(rgb, w, h, png);
  assert(ret);
}

void Phont::load(const std::string &fn) {
  clear();
  std::string png = makemore::slurp(fn);
  pngrgb(png, &w, &h, &rgb);

  dim = 6;
  assert(w == h);
  int dw = w;
  while (dw > 24) {
    assert(dw % 2 == 0);
    dw /= 2;
    --dim;
  }
  assert(dw == 24);
}

void Phont::load(const Partrait &prt) {
  load(prt.rgb, prt.w, prt.h);
}

void Phont::load(const uint8_t *_rgb, unsigned int _w, unsigned int _h) {
  clear();
  w = _w;
  h = _h;
  rgb = new uint8_t[w * h * 3];
  memcpy(rgb, _rgb, w * h * 3);

  dim = 6;
  assert(w == h);
  int dw = w;
  while (dw > 24) {
    assert(dw % 2 == 0);
    dw /= 2;
    --dim;
  }
  assert(dw == 24);
}

Phont::~Phont() {
  clear();
}

Phont::Phont(const Phont &ph) {
  rgb = NULL;
  copy(ph);
}

Phont &Phont::operator =(const Phont &ph) {
  copy(ph);
  return *this;
}

void Phont::copy(const Phont &ph) {
  clear();
  w = ph.w;
  h = ph.h;
  dim = ph.dim;

  if (ph.rgb) {
    rgb = new uint8_t[w * h * 3];
    memcpy(rgb, ph.rgb, w * h * 3);
  }
}

void Phont::reduce() {
  assert(rgb);
  assert(dim < 6);

  ++dim;
  assert(w == h);
  assert(w % 2 == 0);
  w /= 2;
  h /= 2;

  uint8_t *new_rgb = new uint8_t[w * h * 3];
  uint8_t *p = new_rgb;
  unsigned int wh3 = w * h * 3;

  for (unsigned int y = 0; y < h; ++y) {
    for (unsigned int x = 0; x < w; ++x) {
      for (unsigned int c = 0; c < 3; ++c) {
        unsigned int z =
          rgb[(2 * y + 0) * 6 * w + (2 * x + 0) * 3 + c] +
          rgb[(2 * y + 1) * 6 * w + (2 * x + 0) * 3 + c] +
          rgb[(2 * y + 0) * 6 * w + (2 * x + 1) * 3 + c] +
          rgb[(2 * y + 1) * 6 * w + (2 * x + 1) * 3 + c];

        z += 2;
        z >>= 2;

        assert(z >= 0 && z < 256);
        *p++ = z;
      }
    }
  }

  delete[] rgb;
  rgb = new_rgb;
}

void Phont::enlarge() {
  assert(rgb);
  assert(dim > -3);

  --dim;
  unsigned int w0 = w, h0 = h;
  w *= 2;
  h *= 2;

  uint8_t *new_rgb = new uint8_t[w * h * 3];
  uint8_t *p = new_rgb;
  unsigned int wh3 = w * h * 3;

  for (unsigned int y = 0; y < h; ++y) {
    unsigned int y0 = (y >> 1);
    for (unsigned int x = 0; x < w; ++x) {
      unsigned int x0 = (x >> 1);
      for (unsigned int c = 0; c < 3; ++c) {
        *p++ = rgb[y0 * 3 * w0 + x0 * 3 + c];
      }
    }
  }

  delete[] rgb;
  rgb = new_rgb;
}

void Phont::set_dim(int new_dim) {
  while (dim < new_dim)
    reduce();
  while (dim > new_dim)
    enlarge();
  assert(dim == new_dim);
}

Fontasy::Fontasy(unsigned int _w, unsigned int _h) {
  fg = 0xFFFFFF;
  bg = 0x000000;
  fga = 0xFF;
  bga = 0xFF;
  dim = 1;
  x = y = 0;

  w = _w;
  h = _h;
  rgb = new uint8_t[w * h * 3]();

  enc = new Superenc("genc.proj", 1);
  gen = new Supergen("ggen.proj", 1);
  bst = new Supergen("zoom2.proj", 1);
  sty = new Styler("gsty.proj");

  ahoy = new Phont("ahoy.png");
  mork = new Phont("mork.png");
  they = new Phont("they.png");

  base = sty->tag_cholo["base"];
  assert(base);
}

Fontasy::~Fontasy() {
  delete[] rgb;

  delete enc;
  delete gen;
  delete bst;
  delete sty;

  delete mork;
  delete ahoy;
  delete they;
}

void Fontasy::encode(const Phont &phont, double *ctr) {
  const Phont *pp = &phont;
  Phont pt;
  if (phont.dim != 3) {
    pt = phont;
    pt.set_dim(3);
    pp = &pt;
  }

  assert(pp->w == 192 && pp->h == 192);

  Partrait tmp;
  tmp.w = pp->w;
  tmp.h = pp->h;
  tmp.rgb = pp->rgb;

  enc->encode(tmp, ctr);

  tmp.rgb = NULL;

  base->encode(ctr, ctr);
}

void Fontasy::boost(Phont *pp) {
  assert(pp->dim == 3 && pp->w == 192 && pp->h == 192);

  double *buf = new double[pp->w * pp->h * 3];
  rgblab(pp->rgb, pp->w * pp->h * 3, buf);

  Partrait tmp2(768, 768);
  bst->generate(buf, &tmp2);
  delete[] buf;

  pp->load(tmp2);
}

void Fontasy::generate(const double *ctr, Phont *pp) {
  Partrait tmp(192, 192);

  double *tctr = new double[1024];
  base->generate(ctr, tctr);

  for (unsigned int i = 0; i < 1024; ++i) {
    if (tctr[i] > 1.0)
      tctr[i] = 0.0;
    else if (tctr[i] < 0.0)
      tctr[i] = 0.0;
  }

  gen->generate(tctr, &tmp);
  delete[] tctr;

  pp->load(tmp);
  boost(pp);
}

static void namectr(const char *name, double *ctr) {
  const char *p = strchr(name, '_');

  memset(ctr, 0, 1024 * sizeof(double));


  std::string word;
  double m = 0.35;

  if (p) {
    word = std::string(name, p - name);
    namectr(p + 1, ctr);
  } else {
    word = name;
  }

  unsigned int s = 133;
  for (const char *q = word.c_str(); *q; ++q)
    s = s * 17 + *q;
  seedrand(s);

  for (unsigned int i = 0; i < 1024; ++i)
    ctr[i] += randgauss() * m;
}

std::string Fontasy::gennom() {
  std::string a = fontnoms[randuint() % nfontnoms];

  if (randuint() % 4 != 0)
    return a;

  std::string b = fontnoms[randuint() % nfontnoms];
  return a + "_" + b;
}

std::string Fontasy::gennom(const std::string &rel) {
  std::string r;
  const char *p = strchr(rel.c_str(), '_');
  if (p) {
    if (randuint() % 2) {
      r = std::string(rel.c_str(), p);
    } else {
      r = p + 1;
    }
  } else {
    r = rel;
  }

  if (r != rel && randuint() % 32 == 0)
    return r;

  std::string a = r;
  std::string b;
  do {
    b = fontnoms[randuint() % nfontnoms];
  } while (b == a);

  if (randuint() % 2) {
    return a + "_" + b;
  } else {
    return b + "_" + a;
  }
}

void Fontasy::generate(const std::string &name, Phont *pp, double m) {
  if (name == "mork") { *pp = *mork; return; }
  if (name == "ahoy") { *pp = *ahoy; return; }
  if (name == "they") { *pp = *they; return; }

  double *ctr = new double[1024];
  namectr(name.c_str(), ctr);

  for (unsigned int i = 0; i < 1024; ++i)
    ctr[i] *= m;

  generate(ctr, pp);

  delete[] ctr;
}

void Fontasy::to_png(std::string *png) const {
  bool ret = rgbpng(rgb, w, h, png);
  assert(ret);
}

void Fontasy::save(FILE *fp) {
  std::string png;
  to_png(&png);
  makemore::spit(png, fp);
}

void Fontasy::newline() {
  int ch = 3 * (dim > 0 ? (64 >> dim) : (64 << -dim));
  y += ch;
  x = 0;
}
void Fontasy::up() {
  int ch = 3 * (dim > 0 ? (64 >> dim) : (64 << -dim));
  y -= ch;
}
void Fontasy::down() {
  int ch = 3 * (dim > 0 ? (64 >> dim) : (64 << -dim));
  y += ch;
}
void Fontasy::left() {
  int cw = 2 * (dim > 0 ? (64 >> dim) : (64 << -dim));
  x -= cw;
}
void Fontasy::right() {
  int cw = 2 * (dim > 0 ? (64 >> dim) : (64 << -dim));
  x += cw;
}
void Fontasy::smaller() {
  ++dim;
}
void Fontasy::bigger() {
  --dim;
}

void Fontasy::print(Phont *ph, const std::string &_str) {
  const char *str = _str.data();
  unsigned int n = _str.length();
  for (unsigned int i = 0; i < n; ++i) {
    print(ph, (uint8_t)str[i]);
  }
}

void Fontasy::print(Phont *ph, const char *str) {
  while (*str) {
    print(ph, (uint8_t)*str);
    ++str;
  }
}

void Fontasy::print(Phont *ph, uint8_t c) {
  int raw = 0;
  int cw = 2 * (dim > 0 ? (64 >> dim) : (64 << -dim));
  int ch = 3 * (dim > 0 ? (64 >> dim) : (64 << -dim));

  if (c < 0x20) {
    ph = ahoy;
    c += '@';
  } else if (c >= 0x80) {
    c -= 0x80;

    if (c < 0x20) {
      ph = they;
      c += '@';
    } else {
      ph = mork;
    }
  }

  if (ph == ahoy || ph == they || ph == mork) {
    raw = 1;
  }

  assert(c >= 0x20 && c < 0x80);
  assert(dim == ph->dim);

  int cx0 = cw * ((c - 0x20) % 12);
  int cy0 = ch * (int)((c - 0x20) / 12);
  int cx1 = cx0 + cw;
  int cy1 = cy0 + ch;

  if (raw) {
    for (int cy = cy0; cy < cy1; ++cy) {
      for (int cx = cx0; cx < cx1; ++cx) {
        for (int k = 0; k < 3; ++k) {
          int z = ph->rgb[cy * ph->w * 3 + cx * 3 + k];
          int dy = y + cy - cy0;
          int dx = x + cx - cx0;
          rgb[dy * w * 3 + dx * 3 + k] = z;
        }
      }
    }
  } else {
    for (int cy = cy0; cy < cy1; ++cy) {
      for (int cx = cx0; cx < cx1; ++cx) {
        double z = ph->rgb[cy * ph->w * 3 + cx * 3 + 0] +
          ph->rgb[cy * ph->w * 3 + cx * 3 + 1] +
          ph->rgb[cy * ph->w * 3 + cx * 3 + 2];
        z /= 3;
        z = 255 - z;
        z /= 255.0;

        int dy = y + cy - cy0;
        int dx = x + cx - cx0;

        double a = (fga * z + bga * (1.0 - z)) / 255.0;

        double fgr = (fg >> 16) & 0xFF;
        double bgr = (bg >> 16) & 0xFF;
        double r1 = fgr * z + bgr * (1 - z);
        double r0 = rgb[dy * w * 3 + dx * 3 + 0];
        rgb[dy * w * 3 + dx * 3 + 0] = r0 * (1 - a) + r1 * a;

        double fgg = (fg >> 8) & 0xFF;
        double bgg = (bg >> 8) & 0xFF;
        double g1 = fgg * z + bgg * (1 - z);
        double g0 = rgb[dy * w * 3 + dx * 3 + 1];
        rgb[dy * w * 3 + dx * 3 + 1] = g0 * (1 - a) + g1 * a;

        double fgb = (fg >> 0) & 0xFF;
        double bgb = (bg >> 0) & 0xFF;
        double b1 = fgb * z + bgb * (1 - z);
        double b0 = rgb[dy * w * 3 + dx * 3 + 2];
        rgb[dy * w * 3 + dx * 3 + 2] = b0 * (1 - a) + b1 * a;
      }
    }
  }

  x += cw;
}

void Phont::print(const std::string &_str, uint32_t fg, uint32_t bg, uint8_t fga, uint8_t bga, Pic *pic) {
  const char *str = _str.data();
  unsigned int n = _str.length();

  int cw = 2 * (dim > 0 ? (64 >> dim) : (64 << -dim));
  int x = 0, y = 0;
  for (unsigned int i = 0; i < n; ++i) {
    print((uint8_t)str[i], x, y, fg, bg, fga, bga, pic);
    x += cw;
  }
}

void Phont::print(uint8_t c, int x, int y, uint32_t fg, uint32_t bg, uint8_t fga, uint8_t bga, Pic *pic) {
  int raw = 0;
  int cw = 2 * (dim > 0 ? (64 >> dim) : (64 << -dim));
  int ch = 3 * (dim > 0 ? (64 >> dim) : (64 << -dim));

  if (c < 0x20 || c >= 0x80)
    c = 0x7F;
  assert(c >= 0x20 && c < 0x80);

  int cx0 = cw * ((c - 0x20) % 12);
  int cy0 = ch * (int)((c - 0x20) / 12);
  int cx1 = cx0 + cw;
  int cy1 = cy0 + ch;

  for (int cy = cy0; cy < cy1; ++cy) {
    for (int cx = cx0; cx < cx1; ++cx) {
      int dy = y + cy - cy0;
      int dx = x + cx - cx0;
      if (dx < 0 || dx >= pic->w || dy < 0 || dy >= pic->h)
        continue;

#if 0
      double z = rgb[cy * w * 3 + cx * 3 + 0] +
        rgb[cy * w * 3 + cx * 3 + 1] +
        rgb[cy * w * 3 + cx * 3 + 2];
      z /= 3;
      z = 255 - z;
      z /= 255.0;
      double a = (fga * z + bga * (1.0 - z)) / 255.0;
#endif

      double zr = (255.0 - rgb[cy * w * 3 + cx * 3 + 0]) / 255.0;
      double ar = (fga * zr + bga * (1.0 - zr)) / 255.0;
      double fgr = (fg >> 16) & 0xFF;
      double bgr = (bg >> 16) & 0xFF;
      double r1 = fgr * zr + bgr * (1 - zr);
      double r0 = pic->rgb[dy * pic->w * 3 + dx * 3 + 0];
      pic->rgb[dy * pic->w * 3 + dx * 3 + 0] = r0 * (1 - ar) + r1 * ar;

      double zg = (255.0 - rgb[cy * w * 3 + cx * 3 + 1]) / 255.0;
      double ag = (fga * zg + bga * (1.0 - zg)) / 255.0;
      double fgg = (fg >> 8) & 0xFF;
      double bgg = (bg >> 8) & 0xFF;
      double g1 = fgg * zg + bgg * (1 - zg);
      double g0 = pic->rgb[dy * pic->w * 3 + dx * 3 + 1];
      pic->rgb[dy * pic->w * 3 + dx * 3 + 1] = g0 * (1 - ag) + g1 * ag;

      double zb = (255.0 - rgb[cy * w * 3 + cx * 3 + 2]) / 255.0;
      double ab = (fga * zb + bga * (1.0 - zb)) / 255.0;
      double fgb = (fg >> 0) & 0xFF;
      double bgb = (bg >> 0) & 0xFF;
      double b1 = fgb * zb + bgb * (1 - zb);
      double b0 = pic->rgb[dy * pic->w * 3 + dx * 3 + 2];
      pic->rgb[dy * pic->w * 3 + dx * 3 + 2] = b0 * (1 - ab) + b1 * ab;
    }
  }

  x += cw;
}

void Phont::print_all(uint32_t fg, uint32_t bg, Pic *pic) {
  int cw = 2 * (dim > 0 ? (64 >> dim) : (64 << -dim));
  int ch = 3 * (dim > 0 ? (64 >> dim) : (64 << -dim));

  for (int c = 0x20; c < 0x80; ++c) {
    int x = cw * ((c - 0x20) % 12);
    int y = ch * (int)((c - 0x20) / 12);
    print(c, fg, x, y, bg, 0xFF, 0xFF, pic);
  }
}


}

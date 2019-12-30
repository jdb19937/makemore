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

#include <SDL2/SDL.h>

namespace makemore {

struct Phont {
  uint8_t *rgb;
  int dim;
  unsigned int w, h;

  Phont() {
    rgb = NULL;
    w = h = 0;
    dim = 0;
  }

  Phont(const std::string &fn) {
    rgb = NULL;
    w = h = 0;
    dim = 0;
    load(fn);
  }

  void clear() {
    if (rgb) {
      delete[] rgb;
      rgb = NULL;
    }
    w = h = 0;
    dim = 0;
  }

  void load(const std::string &fn) {
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

  void load(const Partrait &prt) {
    clear();
    w = prt.w;
    h = prt.h;
    rgb = new uint8_t[w * h * 3];
    memcpy(rgb, prt.rgb, w * h * 3);

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

  ~Phont() {
    clear();
  }

  Phont(const Phont &ph) {
    rgb = NULL;
    copy(ph);
  }

  Phont &operator =(const Phont &ph) {
    copy(ph);
    return *this;
  }

  void copy(const Phont &ph) {
    clear();
    w = ph.w;
    h = ph.h;
    dim = ph.dim;

    if (ph.rgb) {
      rgb = new uint8_t[w * h * 3];
      memcpy(rgb, ph.rgb, w * h * 3);
    }
  }

  void reduce() {
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

  void enlarge() {
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

  void set_dim(int new_dim) {
    while (dim < new_dim)
      reduce();
    while (dim > new_dim)
      enlarge();
    assert(dim == new_dim);
  }
};

struct PhontSet {
  std::map<int, Phont *> ph;

  PhontSet(const std::string &fn) {
    ph[1] = new Phont(fn);
  }

  PhontSet(const Phont &p) {
    ph[1] = new Phont(p);
  }

  ~PhontSet() {
    for (auto phi : ph) {
      delete phi.second;
    }
  }

  const Phont *get(int d) {
    assert(d <= 6 && d >= -3);
    auto phi = ph.find(d);
    if (phi != ph.end())
      return phi->second;

    phi = ph.find(1);
    assert(phi != ph.end());
    Phont *p = new Phont(*phi->second);
    p->set_dim(d);

    ph[d] = p;
    return p;
  }
};


struct Fontasy {
  Superenc *enc;
  Supergen *gen, *bst;
  Cholo *base;
  Styler *sty;
  PhontSet *mork, *they, *ahoy;
  std::map<std::string, PhontSet*> fmap;

  unsigned int w, h;
  uint8_t *rgb;

  int x, y;
  int dim;
  uint32_t fg, bg;
  uint8_t fga, bga;
  PhontSet *phs;

  Fontasy(unsigned int _w, unsigned int _h) {
    fg = 0xFFFFFF;
    bg = 0x000000;
    fga = 0xFF;
    bga = 0xFF;
    dim = 1;
    x = y = 0;
    phs = NULL;

    w = _w;
    h = _h;
    rgb = new uint8_t[w * h * 3]();

    enc = new Superenc("fenc.proj", 1);
    gen = new Supergen("fgen.proj", 1);
    bst = new Supergen("boost.proj", 1);
    sty = new Styler("fsty.proj");

    fmap["ahoy"] = ahoy = new PhontSet("/home/dan/ftest/ahoy.png");
    fmap["mork"] = mork = new PhontSet("/home/dan/ftest/mork.png");
    fmap["they"] = they = new PhontSet("/home/dan/ftest/they.png");

    base = sty->tag_cholo["base"];
    assert(base);

    phs = they;
  }

  ~Fontasy() {
    delete[] rgb;

    for (auto fi : fmap)
      delete fi.second;

    delete enc;
    delete gen;
    delete bst;
    delete sty;
  }

  void encode(const Phont &phont, double *ctr) {
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

  void boost(Phont *pp) {
    assert(pp->dim == 3 && pp->w == 192 && pp->h == 192);

    double *buf = new double[pp->w * pp->h * 3];
    rgblab(pp->rgb, pp->w * pp->h * 3, buf);

    Partrait tmp2(768, 768);
    bst->generate(buf, &tmp2);
    delete[] buf;

    pp->load(tmp2);
  }

  void generate(const double *ctr, Phont *pp) {
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

  void to_png(std::string *png) const {
    bool ret = rgbpng(rgb, w, h, png);
    assert(ret);
  }

  void save(FILE *fp) {
    std::string png;
    to_png(&png);
    makemore::spit(png, fp);
  }

  void newline() {
    int ch = 3 * (dim > 0 ? (64 >> dim) : (64 << -dim));
    y += ch;
    x = 0;
  }
  void up() {
    int ch = 3 * (dim > 0 ? (64 >> dim) : (64 << -dim));
    y -= ch;
  }
  void down() {
    int ch = 3 * (dim > 0 ? (64 >> dim) : (64 << -dim));
    y += ch;
  }
  void left() {
    int cw = 2 * (dim > 0 ? (64 >> dim) : (64 << -dim));
    x -= cw;
  }
  void right() {
    int cw = 2 * (dim > 0 ? (64 >> dim) : (64 << -dim));
    x += cw;
  }
  void smaller() {
    ++dim;
  }
  void bigger() {
    --dim;
  }

  void print(const std::string &_str) {
    const char *str = _str.data();
    unsigned int n = _str.length();
    for (unsigned int i = 0; i < n; ++i) {
      print((uint8_t)str[i]);
    }
  }

  void print(const char *str) {
    while (*str) {
      print((uint8_t)*str);
      ++str;
    }
  }

  void print(uint8_t c) {
    PhontSet *s = phs;

    int raw = 0;
    int cw = 2 * (dim > 0 ? (64 >> dim) : (64 << -dim));
    int ch = 3 * (dim > 0 ? (64 >> dim) : (64 << -dim));

    if (c == '\b') {
      uint8_t buf[4] = {0x80 + 'a', ' ', 0x80 + 'a', 0};
      print((const char *)buf);
      return;
    }

    if (c == '\n') {
      newline();
      return;
    }

    if (c == 0x80 + 'w') {
      up();
      return;
    }
    if (c == 0x80 + 'a') {
      left();
      return;
    }
    if (c == 0x80 + 's') {
      down();
      return;
    }
    if (c == 0x80 + 'd') {
      right();
      return;
    }

    if (c == 0x80 + '+') {
      bigger();
      return;
    }

    if (c == 0x80 + '-') {
      smaller();
      return;
    }

    if (c < 0x20) {
      s = ahoy;
      c += '@';
    } else if (c >= 0x80) {
      c -= 0x80;

      if (c < 0x20) {
        s = they;
        c += '@';
      } else {
        s = mork;
      }
    }

    if (s == ahoy || s == they || s == mork) {
      raw = 1;
    }

    assert(c >= 0x20 && c < 0x80);

    const Phont *ph = s->get(dim);
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

  void set_phont(const std::string& name, double vdev = 1.0) {
    auto fi = fmap.find(name);
    if (fi != fmap.end()) {
      phs = fi->second;
      return;
    }

    seedrand(1729);
    for (unsigned int i = 0; i < name.length(); ++i)
      seedrand(randuint() + name[i]);

    double *ctr = new double[1024];
    for (int i = 0; i < 1024; ++i)
      ctr[i] = randgauss() * vdev;

    Phont ph;
    generate(ctr, &ph);
    delete[] ctr;

    phs = new PhontSet(ph);

    fmap[name] = phs;
  }
};

}

using namespace makemore;

int main() {
  Display display;
  display.open();

  seedrand();

  Fontasy fon(display.w, display.h);
  double ctr[1024];
  //Phont ph0("/home/dan/ftest/zud_juice.png");
  //fon.encode(ph0, ctr);

  fon.set_phont("notnormal", 1.0);

fon.print("abcdef\nqwerty\nfuthark\n");
char str[256];
sprintf(str, "%chello%caro%c%c%c%cxxx\nbarf\nu%c\"mlaut",
0x80 + '+', 0x80 + '-', 0x80 + 's', 0x80 + 'a', 0x80 + 'a', 0x80 + 'a', 0x80 + 'a');
fon.print( str);

  // fon.save(stdout);

  unsigned int bufn = fon.w * fon.h * 3;
  uint8_t *buf = new uint8_t[bufn];

  bool done = false;
  long blink = 0;

  while (!done) {
    double t = now();
    blink = (long)(t * 2.5) % 2;

    if (blink) {
      memcpy(buf, fon.rgb, bufn);
      fon.print(0x80 + 'W');
    }

    display.update(fon.rgb, fon.w, fon.h);

    if (blink) {
      fon.print('\b');
      memcpy(fon.rgb, buf, bufn);
    }

    display.present();

    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_KEYDOWN) {
        int k = event.key.keysym.sym;
        if (k == SDLK_ESCAPE) {
          done = true;
          break;
        }
        if (k == SDLK_BACKSPACE) {
          fon.print('\b');
          continue;
        }
        if (k == SDLK_RETURN) {
          fon.newline();
          continue;
        }
        if (k == SDLK_UP) { fon.up(); continue; }
        if (k == SDLK_DOWN) { fon.down(); continue; }
        if (k == SDLK_LEFT) { fon.left(); continue; }
        if (k == SDLK_RIGHT) { fon.right(); continue; }
        if ((event.key.keysym.mod & KMOD_CTRL) && (k == 'r' || k == 'R')) { 
          char f[256];
          sprintf(f, "font%u", randuint());
          fon.set_phont(f, 1.0);
          continue;
        }

        if (!(k >= 0 && k < 0x80)) {
          continue;
        }

        if (event.key.keysym.mod & KMOD_CTRL) { 
          if (k >= 'a' && k <= 'z')
            k += 'A' - 'a';
          if (k >= '@' && k < '@' + 0x20) {
            k -= '@';
          } else {
            k = 0x7F;
          }
        }

        if (event.key.keysym.mod & KMOD_SHIFT) {
          if (k >= '0' && k <= '9')
            k = ")!@#$%^&*("[k - '0'];
          else if (k >= 'a' && k <= 'z')
            k = k + 'A' - 'a';
          else if (k == '-') k = '_';
          else if (k == '=') k = '+';
          else if (k == '[') k = '{';
          else if (k == ']') k = '}';
          else if (k == ';') k = ':';
          else if (k == '\'') k = '"';
          else if (k == ',') k = '<';
          else if (k == '.') k = '>';
          else if (k == '/') k = '?';
          else if (k == '\\') k = '|';
          else if (k == '`') k = '~';
        }

        if (event.key.keysym.mod & KMOD_ALT)
          k |= 0x80;

        fon.print((uint8_t)k);
      }
    }
  }

  return 0;
}

#ifndef __MAKEMORE_FONTASY_HH__
#define __MAKEMORE_FONTASY_HH__ 1

#include <string>
#include <map>

namespace makemore {

struct Phont {
  uint8_t *rgb;
  int dim;
  unsigned int w, h;

  Phont();
  Phont(const std::string &fn);
  Phont(const Phont &ph);

  ~Phont();

  void clear();
  void load(const std::string &fn);
  void load(const class Partrait &prt);
  void load(const uint8_t *rgb, unsigned int w, unsigned int h);
  void to_png(std::string *png) const;

  Phont &operator =(const Phont &ph);
  void copy(const Phont &ph);

  void reduce();
  void enlarge();

  void set_dim(int new_dim);

  void print(const std::string &str, uint32_t fg, uint32_t bg, uint8_t fga, uint8_t bga, class Pic *pic);
  void print(uint8_t c, int x, int y, uint32_t fg, uint32_t bg, uint8_t fga, uint8_t bga, class Pic *pic);
  void print_all(uint32_t fg, uint32_t bg, class Pic *pic);
};

struct Fontasy {
  class Superenc *enc;
  class Supergen *gen, *bst;
  class Cholo *base;
  class Styler *sty;
  Phont *mork, *they, *ahoy;

  unsigned int w, h;
  uint8_t *rgb;

  int x, y;
  int dim;
  uint32_t fg, bg;
  uint8_t fga, bga;

  Fontasy(unsigned int _w, unsigned int _h);
  ~Fontasy();

  void encode(const Phont &phont, double *ctr);
  void boost(Phont *pp);
  void generate(const double *ctr, Phont *pp);
  void generate(const std::string &name, Phont *pp, double m = 1.0);
  std::string gennom();
  std::string gennom(const std::string &rel);

  void to_png(std::string *png) const;
  void save(FILE *fp);

  void newline();
  void up();
  void down();
  void left();
  void right();
  void smaller();
  void bigger();

  void print(Phont *ph, const std::string &_str);
  void print(Phont *ph, const char *str);
  void print(Phont *ph, uint8_t c);
};

}
#endif

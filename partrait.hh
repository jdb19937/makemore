#ifndef __MAKEMORE_PARTRAIT_HH__
#define __MAKEMORE_PARTRAIT_HH__ 1

#include <stdlib.h>
#include <string.h>

#include <vector>
#include <string>
#include <set>

#include "point.hh"
#include "triangle.hh"
#include "pose.hh"
#include "hashbag.hh"

namespace makemore {

struct Partrait {
  unsigned int w, h;
  uint8_t *rgb;
  uint8_t *alpha;

  std::vector<std::string> tags;

  Partrait();
  Partrait(unsigned int _w, unsigned int _h);
  Partrait(const Partrait &par) {
    w = par.w;
    h = par.h;

    if (par.rgb) {
      assert(w > 0 && h > 0);
      rgb = new uint8_t[w * h * 3];
      memcpy(rgb, par.rgb, w * h * 3);
    } else {
      rgb = NULL;
      assert(w == 0 && h == 0);
    }

    if (par.alpha) {
      alpha = new uint8_t[w * h * 1];
      memcpy(alpha, par.alpha, w * h * 1);
    } else {
      alpha = NULL;
    }

    tags = par.tags;
  }

  ~Partrait();

  bool empty() const;

  void clear();

  void create(unsigned int _w, unsigned int _h) {
    clear();
    w = _w;
    h = _h;
    assert(w > 0 && h > 0);
    rgb = new uint8_t[w * h * 3];
  }

  void fill_white() {
    assert(rgb);
    memset(rgb, 0xFF, w * h * 3);
  }
  void fill_gray() {
    assert(rgb);
    memset(rgb, 0x80, w * h * 3);
  }
  void fill_black() {
    assert(rgb);
    memset(rgb, 0, w * h * 3);
  }
  void fill_blue() {
    assert(rgb);
    unsigned int wh3 = w * h * 3;
    memset(rgb, 0, wh3);
    for (unsigned int j = 2; j < wh3; j += 3)
      rgb[j] = 255;
  }

  void load(const std::string &fn);
  void save(const std::string &fn) const;
  void save(FILE *) const;
  void to_png(std::string *png) const;
  void from_png(const std::string &png);

  void warp(Partrait *to) const;
  void warpover(Partrait *to) const;

  Pose get_pose() const;
  bool has_pose() const;
  void set_pose(const Pose &);

  bool has_tag(const std::string &q) const {
    for (auto tag : tags)
      if (tag == q)
         return true;
    return false;
  }

  double get_tag(const std::string &k, double dv) const {
    std::string kc = k + ":";
    std::vector<std::string> new_tags;
    for (auto tag : tags)
      if (!strncmp(tag.c_str(), kc.c_str(), kc.length()))
        return strtod(tag.c_str() + kc.length(), NULL);
    return dv;
  }

  std::string get_tag(const std::string &k, const std::string &dv = "") const {
    std::string kc = k + ":";
    std::vector<std::string> new_tags;
    for (auto tag : tags)
      if (!strncmp(tag.c_str(), kc.c_str(), kc.length()))
        return tag.c_str() + kc.length();
    return dv;
  }

  void set_tag(const std::string &k, double v) {
    char buf[256];
    sprintf(buf, "%lf", v);
    set_tag(k, std::string(buf));
  }

  void set_tag(const std::string &k, long v) {
    char buf[256];
    sprintf(buf, "%ld", v);
    set_tag(k, std::string(buf));
  }

  void set_tag(const std::string &k, const std::string &v) {
    std::string kc = k + ":";
    std::vector<std::string> new_tags;
    for (auto tag : tags)
      if (strncmp(tag.c_str(), kc.c_str(), kc.length()))
        new_tags.push_back(tag);
    new_tags.push_back(kc + v);
    tags = new_tags;
  }

  void bag_tags(Hashbag *hb) {
    for (auto tag : tags) {
      if (!strchr(tag.c_str(), ':')) {
        hb->add(tag.c_str());
      }
    }
  }

  Triangle get_mark() const;
  bool has_mark() const;
  void set_mark(const Triangle &);

  Triangle get_auto_mark() const;
  bool has_auto_mark() const;
  void set_auto_mark(const Triangle &);

  void encudub(double *cubuf) const;
  void reflect();
  bool read_ppm(FILE *);
  void write_ppm(FILE *);

  void jitter(unsigned int z = 1);

  void make_sketch(double *sketch, bool bw = false) const;

  void erase_bg(const Partrait &mask);
  void replace_bg(const Partrait &mask, const Partrait &bg);
};

}

#endif

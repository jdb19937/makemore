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

    tags = par.tags;
  }

  ~Partrait();

  bool empty() const;

  void clear();
  void load(const std::string &fn);
  void save(const std::string &fn) const;
  void save(FILE *) const;
  void to_png(std::string *png) const;

  void warp(Partrait *to) const;
  void warpover(Partrait *to) const;

  Pose get_pose() const;
  bool has_pose() const;
  void set_pose(const Pose &);

  bool has_tag(const std::string &q) {
    for (auto tag : tags)
      if (tag == q)
         return true;
    return false;
  }

  double get_tag(const std::string &k, double dv) {
    std::string kc = k + ":";
    std::vector<std::string> new_tags;
    for (auto tag : tags)
      if (!strncmp(tag.c_str(), kc.c_str(), kc.length()))
        return strtod(tag.c_str() + kc.length(), NULL);
    return dv;
  }

  std::string get_tag(const std::string &k, const std::string &dv = "") {
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

  void make_sketch(double *sketch);
};

}

#endif

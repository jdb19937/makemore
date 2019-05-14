#ifndef __MAKEMORE_PARTRAIT_HH__
#define __MAKEMORE_PARTRAIT_HH__ 1

#include <vector>
#include <string>
#include <set>

#include "point.hh"
#include "triangle.hh"
#include "pose.hh"

namespace makemore {

struct Partrait {
  unsigned int w, h;
  uint8_t *rgb;

  std::vector<std::string> tags;

  Partrait();
  Partrait(unsigned int _w, unsigned int _h);
  ~Partrait();

  bool empty() const;

  void clear();
  void load(const std::string &fn);
  void save(const std::string &fn) const;
  void save(FILE *) const;
  void to_png(std::string *png) const;

  void warp(Partrait *to) const;

  Pose get_pose() const;
  bool has_pose() const;
  void set_pose(const Pose &);

  Triangle get_mark() const;
  bool has_mark() const;
  void set_mark(const Triangle &);
};

}

#endif

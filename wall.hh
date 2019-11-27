#ifndef __MAKEMORE_WALL_HH__
#define __MAKEMORE_WALL_HH__ 1

#include <stdio.h>

#include <string>
#include <vector>

#include "strutils.hh"

namespace makemore {

struct Wall {
  strvec posts;
  std::vector<strvec> replies;

  Wall() {
  }

  ~Wall() {
  }

  void clear() {
    posts.clear();
    replies.clear();
  }

  void load(const std::string &fn);
  void load(FILE *);
  void save(const std::string &fn);

  bool erase(unsigned int i, const std::string &hash);
  bool erase(unsigned int i);
  void truncate(unsigned int m);
};

}

#endif

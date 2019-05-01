#ifndef __MAKEMORE_PROJECT_HH__
#define __MAKEMORE_PROJECT_HH__ 1

#include "layout.hh"
#include "topology.hh"
#include "multitron.hh"

#include <string>
#include <map>

namespace makemore {

struct Project {
  std::string dir;
  std::map<std::string, std::string> config;
  unsigned int mbn;

  Project(const std::string &_dir, unsigned int _mbn = 1);
};

}

#endif

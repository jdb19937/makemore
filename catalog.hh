#ifndef __MAKEMORE_CATALOG_HH__
#define __MAKEMORE_CATALOG_HH__ 1

#include <string>
#include <vector>

#include "partrait.hh"

namespace makemore {

struct Catalog {
  std::vector<std::string> fn;

  Catalog();
  Catalog(const std::string &dir);
  ~Catalog();

  void add_dir(const std::string &dir);
  void pick(Partrait *par, unsigned int n = 1, bool randreflect = false);
};

};

#endif

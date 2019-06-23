#ifndef __MAKEMORE_STYLER_HH__
#define __MAKEMORE_STYLER_HH__ 1

#include <string.h>
#include <stdlib.h>

#include <string>
#include <map>

#include "project.hh"
#include "cholo.hh"

namespace makemore {

struct Styler : Project {
  double *tmp;
  unsigned int dim;
  std::map<std::string, Cholo*> tag_cholo;

  Styler(const std::string &dir);

  ~Styler() {
    delete[] tmp;
  }

  void add_cholo(const std::string &tag, const std::string &fn) {
    assert(tag_cholo.find(tag) == tag_cholo.end());
    tag_cholo[tag] = new Cholo(fn, dim);
  }

  void encode(const double *ctr, Parson *prs);
  void generate(const Parson &prs, double *ctr, unsigned int m = 1);
};

}

#endif

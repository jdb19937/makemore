#ifndef __MAKEMORE_SUPERENC_HH__
#define __MAKEMORE_SUPERENC_HH__ 1

#include "layout.hh"
#include "multitron.hh"
#include "project.hh"
#include "parson.hh"
#include "mapfile.hh"
#include "partrait.hh"
#include "cholo.hh"
#include "styler.hh"
#include "supertron.hh"

#include <vector>
#include <string>
#include <map>

namespace makemore {

struct Superenc : Project {
  bool ctract;

  Layout *inplay, *ctrlay;
  Layout *encinlay;

  Mapfile *encmap;

  Supertron *enc;

  double *cuencin, *cuencinp;
  double *ctrbuf, *inpbuf;

  unsigned int rounds;

  Superenc(const std::string &_dir, unsigned int _mbn);
  ~Superenc();

  void report(const char *prog);
  void load();
  void save();

  void encode(const Partrait &prt, double *ctr);
  void burn(const class Supergen &gen, double nu);
};

}

#endif

#ifndef __MAKEMORE_SUPERDIS_HH__
#define __MAKEMORE_SUPERDIS_HH__ 1

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

struct Superdis : Project {
  Layout *inplay;
  Mapfile *dismap;
//  Mapfile *clsmap;
//  Supertron *cls;
  Supertron *dis;

//  double *cuclsin, *cuclstgt;
  double *cudisin, *cudistgt;
  double *inpbuf;

  unsigned int rounds;

  Superdis(const std::string &_dir, unsigned int _mbn);
  ~Superdis();

  void report(const char *prog);
  void load();
  void save();

  double score(const class Supergen *gen);
  double score(const class Partrait &prt);
//  void classify(const class Partrait &prt, double *clsbuf);
  void burn(double sc, double nu);
  void observe(const class Partrait *prt0, class Superenc *enc, class Supergen *gen, const class Partrait *prt1, double nu);
};

}

#endif

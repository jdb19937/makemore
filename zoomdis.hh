#ifndef __MAKEMORE_ZOOMDIS_HH__
#define __MAKEMORE_ZOOMDIS_HH__ 1

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

struct Zoomdis : Project {
  Layout *inplay;
  Mapfile *dismap;
  Supertron *dis;

  double *cudisin, *cudistgt;
  double *inpbuf;

  unsigned int rounds;

  Zoomdis(const std::string &_dir, unsigned int _mbn);
  ~Zoomdis();

  void report(const char *prog);
  void load();
  void save();

  double score(const class Zoomgen *gen);
  double score(const class Partrait &prt);
  void burn(double sc, double nu);
  void observe(const class Partrait *prt0, class Zoomgen *gen, const class Partrait *prt1, double nu);
};

}

#endif

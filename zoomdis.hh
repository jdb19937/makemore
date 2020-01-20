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

  double score(const class Zoomgen *gen, double noise = 0);
  double score(const class Partrait &prt, double noise = 0);
  void burnreal(double nu);
  void burnfake(double nu);
  void testfake();
  void observe(const class Partrait *prt0, class Zoomgen *gen, const class Partrait *prt1, double nu);
};

}

#endif

#ifndef __MAKEMORE_ENHANCER_HH__
#define __MAKEMORE_ENHANCER_HH__ 1

#include "layout.hh"
#include "multitron.hh"
#include "vocab.hh"
#include "project.hh"
#include "script.hh"
#include "convo.hh"
#include "parson.hh"
#include "mapfile.hh"
#include "styler.hh"
#include "supertron.hh"

#include <vector>
#include <string>
#include <map>

namespace makemore {

struct Enhancer : Project {
  Mapfile *genmap, *dismap;

  Supertron *gen, *dis;

  double *cugenin, *cudisin, *cudisfin, *cudistgt, *cugentgt;

  unsigned int rounds;

  Enhancer(const std::string &_dir);
  ~Enhancer();

  void report(const char *prog);
  void load();
  void save();

  void observe(const double *cusamp, double nu);
  void generate(const class Partrait *spic, class Partrait *tpic = NULL);

#if 0
  void burn(double pi, class Zoomdis *dis, const class Partrait *pic, double ganlevel);
  void burn(double pi, const Partrait &);

  double disscore(const class Zoomgen *gen, double noise = 0);
  double disscore(const class Partrait &prt, double noise = 0);
  void disburnreal(double nu);
  void disburnfake(double nu);
  void distestfake();
  void disobserve(const class Partrait *prt0, class Zoomgen *gen, const class Partrait *prt1, double nu);
#endif
};

}

#endif

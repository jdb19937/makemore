#ifndef __MAKEMORE_AUTOMASKER_HH__
#define __MAKEMORE_AUTOMASKER_HH__ 1

#include "layout.hh"
#include "topology.hh"
#include "multitron.hh"
#include "vocab.hh"
#include "project.hh"
#include "script.hh"
#include "convo.hh"
#include "parson.hh"
#include "mapfile.hh"
#include "cholo.hh"

#include <vector>
#include <string>
#include <map>

namespace makemore {

struct Automasker : Project {
  Layout *mskinlay, *mskoutlay;
  Topology *msktop;
  Mapfile *mskmap;
  Tron *msk;

  double *tgtbuf;
  double *cumsktgt, *cumskin;
  unsigned int rounds;

  Automasker(const std::string &_dir);
  ~Automasker();

  void report(const char *prog);
  void load();
  void save();

  void observe(const class Partrait &prt, double mu);
  void automask(class Partrait *prtp);
};

}

#endif

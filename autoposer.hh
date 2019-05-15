#ifndef __MAKEMORE_AUTOPOSER_HH__
#define __MAKEMORE_AUTOPOSER_HH__ 1

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

struct Autoposer : Project {
  Layout *seginlay, *segoutlay;

  Topology *segtop;
  Mapfile *segmap;
  Tron *seg;

  double *cusegtgt, *cusegin;
  unsigned int rounds;

  Autoposer(const std::string &_dir);
  ~Autoposer();

  void report(const char *prog);
  void load();
  void save();

  void observe(const class Partrait &par, double mu);
  void autopose(class Partrait *parp);
};

}

#endif

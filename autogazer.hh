#ifndef __MAKEMORE_AUTOGAZER_HH__
#define __MAKEMORE_AUTOGAZER_HH__ 1

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

struct Autogazer : Project {
  Layout *gazinlay, *gazoutlay;

  Topology *gaztop;
  Mapfile *gazmap;
  Tron *gaz;

  double *cugaztgt, *cugazin;
  unsigned int rounds;

  Autogazer(const std::string &_dir);
  ~Autogazer();

  void report(const char *prog);
  void load();
  void save();

  void observe(const class Partrait &par, double mu);
  void autogaze(class Partrait *parp);
};

}

#endif

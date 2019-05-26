#ifndef __MAKEMORE_IMPDIS_HH__
#define __MAKEMORE_IMPDIS_HH__ 1

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

struct Impdis : Project {
  bool focus;
  double decay;

  Layout *tgtlay;

  Topology *imptop, *distop;
  Mapfile *impmap, *dismap;

  Tron *imp, *dis;

  double *cuimpin, *cuimptgt;
  double *cudisin, *cudistgt, *cuimpfin;

  double *tgtbuf, *inbuf;

  double *cutgtlayx, *cutgtlayy;

  unsigned int rounds;

  Impdis(const std::string &_dir, unsigned int _mbn);
  ~Impdis();

  void report(const char *prog);
  void load();
  void save();

  void burn(double pi);
  void observe(double xi);
};

}

#endif

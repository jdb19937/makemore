#ifndef __MAKEMORE_GENERATOR_HH__
#define __MAKEMORE_GENERATOR_HH__ 1

#include "layout.hh"
#include "topology.hh"
#include "multitron.hh"
#include "vocab.hh"
#include "project.hh"
#include "script.hh"
#include "convo.hh"
#include "parson.hh"
#include "mapfile.hh"
#include "styler.hh"
#include "zone.hh"

#include <vector>
#include <string>
#include <map>

namespace makemore {

struct Generator : Project {
  bool ctract;
  bool focus;
  bool is_rgb;
  int ctxtype;

  Layout *tgtlay, *ctrlay, *ctxlay;
  Layout *geninlay;

  Topology *gentop;
  Mapfile *genmap;

  Tron *gen;
  Zone *zone;

  double *cugenin, *cugentgt, *cugenfin;
  double *ctxbuf, *ctrbuf, *tgtbuf;

  double *cutgtlayx, *cutgtlayy;

  unsigned int rounds;

  Generator(const std::string &_dir, unsigned int _mbn = 1);
  ~Generator();

  void report(const char *prog);
  void load();
  void save();

  void scramble(double dev = 1.0);
  void generate(const class Parson &prs, class Partrait *prt, class Styler *sty = NULL, bool bp = false);
  // void burn(double nu, double pi);
};

}

#endif

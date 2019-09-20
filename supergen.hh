#ifndef __MAKEMORE_SUPERGEN_HH__
#define __MAKEMORE_SUPERGEN_HH__ 1

#include "layout.hh"
#include "multitron.hh"
#include "vocab.hh"
#include "project.hh"
#include "script.hh"
#include "convo.hh"
#include "parson.hh"
#include "mapfile.hh"
#include "styler.hh"
#include "zone.hh"
#include "supertron.hh"

#include <vector>
#include <string>
#include <map>

namespace makemore {

struct Supergen : Project {
  bool ctract;
  bool focus;
  bool is_rgb;
  int ctxtype;

  Layout *tgtlay, *ctrlay, *ctxlay;
  Layout *geninlay;

  Mapfile *genmap;

  Supertron *gen;
  Zone *zone;

  double *cugenin, *cugentgt, *cugenfin;
  double *ctxbuf, *ctrbuf, *tgtbuf, *buf;

  double *cutgtlayx, *cutgtlayy;

  unsigned int rounds;

  Supergen(const std::string &_dir, unsigned int _mbn = 1);
  ~Supergen();

  void report(const char *prog);
  void load();
  void save();

  void scramble(double dev = 1.0);
  void generate(const double *ctr, class Partrait *prt, class Styler *sty = NULL, bool bp = false);
  void burn(const class Partrait &prt, double pi);
  void burn(const class Partrait &prt, double pi, class Superdis *dis);

};

}

#endif

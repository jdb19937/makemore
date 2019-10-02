#ifndef __MAKEMORE_SNDGEN_HH__
#define __MAKEMORE_SNDGEN_HH__ 1

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
#include "soundpic.hh"

#include <vector>
#include <string>
#include <map>

namespace makemore {

struct Sndgen : Project {
  Layout *tgtlay, *ctrlay, *ctxlay;
  Layout *geninlay;

  Mapfile *genmap;
  Supertron *gen;

  double *cugenin, *cugentgt, *cugenfin;
  double *ctxbuf, *ctrbuf, *tgtbuf, *buf;

  unsigned int rounds;

  Sndgen(const std::string &_dir, unsigned int _mbn = 1);
  ~Sndgen();

  void report(const char *prog);
  void load();
  void save();

  void scramble(double dev = 1.0);
  void generate(const double *ctr, Soundpic *sndpic, bool bp = false);
  void burn(const Soundpic &sndpic, double pi);
};

}

#endif

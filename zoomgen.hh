#ifndef __MAKEMORE_ZOOMGEN_HH__
#define __MAKEMORE_ZOOMGEN_HH__ 1

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

struct Zoomgen : Project {
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

  Zoomgen(const std::string &_dir, unsigned int _mbn = 1);
  ~Zoomgen();

  void report(const char *prog);
  void load();
  void save();

  void generate(const class Partrait &, class Partrait *outpic = NULL);
  void burn(double pi, class Zoomdis *dis);
  void burn(double pi, const Partrait &);

};

}

#endif

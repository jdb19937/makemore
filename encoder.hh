#ifndef __MAKEMORE_ENCODER_HH__
#define __MAKEMORE_ENCODER_HH__ 1

#include "layout.hh"
#include "topology.hh"
#include "multitron.hh"
#include "project.hh"
#include "parson.hh"
#include "mapfile.hh"
#include "partrait.hh"
#include "cholo.hh"
#include "styler.hh"

#include <vector>
#include <string>
#include <map>

namespace makemore {

struct Encoder : Project {
  bool ctract;

  Layout *tgtlay, *ctrlay;
  Layout *encinlay;

  Topology *enctop;
  Mapfile *encmap;

  Tron *enc;

  double *cuencin, *cuenctgt;
  double *ctrbuf, *tgtbuf;

  unsigned int rounds;

  Encoder(const std::string &_dir, unsigned int _mbn);
  ~Encoder();

  void report(const char *prog);
  void load();
  void save();

  void encode(const Partrait &prt, class Parson *prs, class Styler *sty = NULL);
  void burn(const Partrait &prt, class Generator *gen, double nu, double pi);
};

}

#endif

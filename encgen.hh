#ifndef __MAKEMORE_ENCGEN_HH__
#define __MAKEMORE_ENCGEN_HH__ 1

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

struct Encgen : Project {
  bool focus;
  bool ctract;
  double decay;

  Layout *tgtlay, *ctrlay, *ctxlay;
  Layout *encinlay, *geninlay;

  Topology *enctop, *gentop;
  Mapfile *encmap, *genmap;

  Tron *enc, *gen;

  double *cuinencin, *cuingenin;
  double *cuencin, *cuenctgt;
  double *cugenin, *cugentgt, *cugenfin;

  double *ctxbuf, *ctrbuf, *tgtbuf;

  double *cutgtlayx, *cutgtlayy;

  unsigned int rounds;

  Encgen(const std::string &_dir, unsigned int _mbn);
  ~Encgen();

  void report(const char *prog);
  void load();
  void save();

  void scramble(double dev = 1.0);
  void generate();
  void encode();
  void burn(double nu, double pi);
};

}

#endif

#ifndef __MAKEMORE_IMPROVER_HH__
#define __MAKEMORE_IMPROVER_HH__ 1

#include <stdio.h>

#include "layout.hh"
#include "topology.hh"
#include "multitron.hh"
#include "project.hh"

#include <string>
#include <map>

namespace makemore {

struct Improver : Project {
  bool do_zoom;
  unsigned int lowoff;
  double shrinkage;

  Layout *tgtlay, *ctrlay, *ctxlay, *outlay;
  Layout *encinlay, *geninlay;

  Topology *enctop, *gentop;

  Multitron *enc, *gen;
  Tron *encpass, *genpass;
  Tron *encgen, *genenc;

  double *cuencin, *cuenctgt;
  double *cugenin, *cugentgt;
  double *realctr, *fakectr, *morectr, *fakectx;

  uint8_t *bctxbuf, *btgtbuf, *boutbuf, *bsepbuf, *bctrbuf;
  uint16_t *sadjbuf;
  double *ctxbuf, *ctrbuf;
  double *tgtbuf, *sepbuf;
  double *outbuf, *adjbuf;

  double *cutgtlayx, *cutgtlayy;

  unsigned int rounds;

  Improver(const char *_dir, unsigned int _mbn = 1);
  virtual ~Improver();

  void burn(double nu, double pi);

  void scramble(double mean, double dev);


  void report(const char *prog, FILE *outfp = stderr);
  void load();
  void save();
};

}

#endif

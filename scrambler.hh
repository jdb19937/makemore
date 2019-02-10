#ifndef __MAKEMORE_SCRAMBLER_HH__
#define __MAKEMORE_SCRAMBLER_HH__ 1

#include "layout.hh"
#include "topology.hh"
#include "multitron.hh"

#include <string>
#include <map>

namespace makemore {

struct Scrambler {
  std::string dir;
  std::map<std::string, std::string> config;

  unsigned int mbn;
  bool do_zoom;
  unsigned int lowoff;
  double shrinkage;

  Layout *tgtlay, *ctrlay, *ctxlay, *outlay;
  Layout *encinlay, *geninlay;

  Topology *enctop, *gentop;

  Multitron *enc, *gen, *genbak;
  Tron *encpass, *genpass;
  Tron *encgen, *genenc;

  double *cuencin, *cuenctgt;
  double *cugenin, *cugentgt;
  double *realctr, *fakectr, *morectr;

  uint8_t *bctxbuf, *btgtbuf, *boutbuf, *bsepbuf;
  double *ctxbuf;
  double *tgtbuf, *sepbuf;
  double *outbuf;

  unsigned int rounds;

  Scrambler(const char *_dir, unsigned int _mbn);
  ~Scrambler();

  void load_ctxtgt(FILE *infp);
  void present(double nu, double mu, double xi);

  void load_ctx(FILE *infp);
  void generate(double dev, unsigned int reps = 1);
  void regenerate();
  void passgenerate();

  void separate();
  void reconstruct();


  void report(const char *prog);
  void load();
  void save();
};

}

#endif

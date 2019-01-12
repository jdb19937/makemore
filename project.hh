#ifndef __MAKEMORE_PROJECT_HH__
#define __MAKEMORE_PROJECT_HH__ 1

#include "layout.hh"
#include "topology.hh"
#include "multitron.hh"

#include <string>
#include <map>

struct Project {
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

  uint8_t *bctxbuf, *btgtbuf, *boutbuf, *bsepbuf, *bctrbuf, *badjbuf;
  double *ctxbuf, *ctrbuf;
  double *tgtbuf, *sepbuf;
  double *outbuf, *adjbuf;

  unsigned int rounds;

  Project(const char *_dir, unsigned int _mbn);
  ~Project();

  void load_ctxtgt(FILE *infp);
  void present(double nu, double mu, double xi, double tau);

  void load_ctx(FILE *infp);
  void generate(unsigned int reps = 1);
  void regenerate();
  void passgenerate();


  void burnmask(double nu);
  void reencode();
  void separate();
  void reconstruct();

  void scramble(double mean, double dev);


  void report(const char *prog);
  void load();
  void save();

  void encode_ctx();
  void encode_ctr();
  void encode_adj();
  void encode_tgt();
  void encode_out();
};

#endif

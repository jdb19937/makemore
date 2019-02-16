#ifndef __MAKEMORE_CONFAB_HH__
#define __MAKEMORE_CONFAB_HH__ 1

#include "layout.hh"
#include "topology.hh"
#include "multitron.hh"
#include "vocab.hh"

#include <string>
#include <map>

namespace makemore {

struct Confab {
  std::string dir;
  std::map<std::string, std::string> config;

  unsigned int mbn;

  Layout *tgtlay, *ctrlay, *ctxlay, *outlay;
  Layout *encinlay, *geninlay;

  Topology *enctop, *gentop, *distop;

  Multitron *enc, *gen, *dis;
  Tron *encpass, *genpass;
  Tron *encgen, *genenc, *gendis;

  double *cuencin, *cuenctgt, *cudistgt;
  double *cugenin, *cugentgt, *cudisin;
  double *realctr, *fakectr, *morectr, *distgt, *fakectx;

  uint8_t *bctxbuf, *btgtbuf, *boutbuf, *bsepbuf, *bctrbuf;
  uint16_t *sadjbuf;
  double *ctxbuf, *ctrbuf;
  double *tgtbuf, *sepbuf;
  double *outbuf, *adjbuf;

  unsigned int rounds;

  Vocab vocab;

  Confab(const char *_dir, unsigned int _mbn);
  ~Confab();

  void generate(unsigned int reps = 1);
  void regenerate();
  void passgenerate();


  void burn(double nu, double pi);
  void reencode();
  void scramble(double mean, double dev);


  void report(const char *prog);
  void load();
  void save();

  void load_ctxtgt(FILE *infp);
  void load_ctx(FILE *infp);

  void encode_ctx();
  void encode_ctr();
  void encode_adj();
  void encode_tgt();
  void encode_out();
};

}

#endif

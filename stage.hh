#ifndef __MAKEMORE_STAGE_HH__
#define __MAKEMORE_STAGE_HH__ 1

#include <stdio.h>

#include "layout.hh"
#include "topology.hh"
#include "multitron.hh"
#include "project.hh"

#include <string>
#include <map>

namespace makemore {

struct Stage : Project {
  bool do_zoom;
  unsigned int lowoff;
  double shrinkage;

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

  double *cutgtlayx, *cutgtlayy;

  unsigned int rounds;

  Stage(const char *_dir, unsigned int _mbn = 1);
  virtual ~Stage();

  void load_ctxtgt(FILE *infp);
  void train_recombine(double yo, double wu, unsigned int js);
  void train_scramble(double yo, double wu);
  void train_fidelity(double nu, double pi, double dcut);
  void train_judgement(double mu, double dcut);
  void train_creativity(double xi, double dcut);

  double encgenerr();

  void load_ctx(FILE *infp);
  void generate(unsigned int reps = 1);
  void regenerate();
  void passgenerate();


  void burn(double nu, double pi);
  void condition(double yo, double wu);
  void reencode(bool force);
  void separate();
  void reconstruct();

  void scramble(double mean, double dev);


  void report(const char *prog, FILE *outfp = stderr);
  void load();
  void save();

  void encode_ctx();
  void encode_ctr();
  void encode_adj();
  void encode_tgt();
  void encode_out();
};

}

#endif

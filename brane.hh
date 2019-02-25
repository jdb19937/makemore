#ifndef __MAKEMORE_BRANE_HH__
#define __MAKEMORE_BRANE_HH__ 1

#include "layout.hh"
#include "topology.hh"
#include "multitron.hh"
#include "vocab.hh"
#include "project.hh"
#include "rule.hh"
#include "script.hh"
#include "convo.hh"
#include "parson.hh"

#include <string>
#include <map>

namespace makemore {

struct Brane : Project {
  Script rules;

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

  Brane(const char *_dir, unsigned int _mbn);
  ~Brane();

  void generate();
  void regenerate();
  void passgenerate();


  void _burn(double nu, double pi);
  void condition(double yo, double wu);
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

  void _ask(const Hashbag &tags, const Convo &req, Shibbomore *rsp, unsigned int rspn);

  std::string ask(const Parson *who, const std::string &req);
  void burn(double pi);
};

}

#endif

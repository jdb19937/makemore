#ifndef __MAKEMORE_PIPELINE_HH__
#define __MAKEMORE_PIPELINE_HH__ 1

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <vector>

#include "layout.hh"
#include "stage.hh"
#include "project.hh"
#include "judge.hh"

namespace makemore {

struct Pipeline : Project {
  std::vector<Stage*> stages;
  Judge *judge;

  Layout *ctrlay, *adjlay;
  const Layout *outlay, *ctxlay;

  double *ctrbuf, *adjbuf, *outbuf, *ctxbuf, *tmpbuf;
  uint32_t tgtlock, ctrlock;

  Pipeline(const char *_dir, unsigned int _mbn = 1);
  void _setup();
  void _add_stage(Stage *);
  virtual ~Pipeline();

  Stage *initial();
  Stage *final();

  void fix(unsigned int iters, double blend);
  void reencode();
  void burn(uint32_t which, double nu, double pi);
  void burngen(double pi);
  void burnenc(double nu);
  void generate();
  void recombine();
  void uptarget();
  void retarget();
  void readjust();
  void autolign(unsigned int iters = 64, int dzoom = 0);

  void load();
  void save();

  void scramble(double mean, double dev);

  void save_ctx_bytes(uint8_t *bbuf);
  void load_ctx_bytes(const uint8_t *bbuf);
  bool load_ctx_bytes(FILE *infp);
  void load_out_bytes(const uint8_t *bbuf);
  bool load_out_bytes(FILE *infp);
  void load_out(const double *buf);

  void report(const char *prog, FILE *outfp = stderr);


  void generate(class Parson **parsons, unsigned int nparsons, long min_age = 0);
  void generate(class Parson *parson, long min_age = 0);
  void generate(class Org *org, long min_age = 0);

  void burnenc(Parson **parsons, unsigned int nparsons, double nu);
  void burngen(Parson **parsons, unsigned int nparsons, double wu);
  void burnencgen(Parson **parsons, unsigned int nparsons, double nu, double pi);
  void discrim(Parson **parsons, unsigned int nparsons, double yo);
};

}

#endif

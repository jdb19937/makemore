#ifndef __MAKEMORE_PIPELINE_HH__
#define __MAKEMORE_PIPELINE_HH__ 1

#include <stdlib.h>
#include <stdint.h>

#include <vector>

#include "layout.hh"
#include "project.hh"

struct Pipeline {
  unsigned int mbn;
  std::vector<Project*> stages;

  Layout *ctrlay, *adjlay;
  const Layout *outlay, *ctxlay;

  double *ctrbuf, *adjbuf, *outbuf, *ctxbuf;
  uint32_t tgtlock, ctrlock;

  Pipeline(unsigned int _mbn);
  void _setup();
  void add_stage(Project *);
  ~Pipeline();

  Project *initial();
  Project *final();

  void fix(unsigned int iters, double blend);
  void reencode();
  void burn(uint32_t which, double nu, double pi);
  void generate();
  void uptarget();
  void retarget();
  void readjust();

  void load();
  void save();

  void scramble(double mean, double dev);

  void save_ctx_bytes(uint8_t *bbuf);
  void load_ctx_bytes(const uint8_t *bbuf);
  bool load_ctx_bytes(FILE *infp);
  void load_out_bytes(const uint8_t *bbuf);
  bool load_out_bytes(FILE *infp);
  void load_out(const double *buf);

  void report(const char *prog);
};

#endif

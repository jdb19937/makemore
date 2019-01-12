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
  uint8_t *bctrbuf, *badjbuf, *boutbuf, *bctxbuf;

  Pipeline(unsigned int _mbn);
  void _setup();
  void add_stage(Project *);
  ~Pipeline();

  Project *initial();
  Project *final();

  void reencode(uint32_t which);
  void burnmask(uint32_t which, double nu);
  void generate();
  void retarget();
  void readjust();

  void load();
  void save();

  void scramble(double mean, double dev);

  void encode_out();
  void encode_adj();
  void encode_ctx();
  void encode_ctr();

  bool load_ctx(FILE *);
  bool load_ctx(const uint8_t *);

  bool load_ctr(FILE *);
  bool load_ctr(const uint8_t *);

  bool load_out(FILE *);
  bool load_out(const uint8_t *);

  bool load_adj(FILE *);
  bool load_adj(const uint8_t *);
};

#endif

#ifndef __MAKEMORE_JUDGE_HH__
#define __MAKEMORE_JUDGE_HH__ 1

#include <stdio.h>

#include "layout.hh"
#include "topology.hh"
#include "multitron.hh"
#include "project.hh"

#include <string>
#include <map>

namespace makemore {

struct Judge : Project {
  unsigned int lowoff;
  Layout *tgtlay, *ctxlay;

  Topology *distop;

  Multitron *dis;

  double *cudistgt;
  double *cudisin;
  double *cudisfin;
  double *disfin;

  double *ctxbuf;
  double *tgtbuf;

  unsigned int rounds;

  Judge(const char *_dir, unsigned int _mbn = 1);
  virtual ~Judge();

  void burn(double yo, double *modtgtbuf = NULL, bool update_stats = true);
  void load();
  void save();

  void report(const char *prog, FILE *outfp);
};

}

#endif

#define __MAKEMORE_JUDGE_CC__ 1

#include <string>
#include <netinet/in.h>

#include "judge.hh"
#include "project.hh"
#include "cudamem.hh"
#include "tron.hh"
#include "multitron.hh"
#include "twiddle.hh"
#include "closest.hh"

namespace makemore {

using namespace std;

Judge::Judge(const char *_dir, unsigned int _mbn) : Project(_dir, _mbn) {
  char ctxlayfn[4096];
  sprintf(ctxlayfn, "%s/context.lay", _dir);
  ctxlay = new Layout;
  ctxlay->load_file(ctxlayfn);

  char tgtlayfn[4096];
  sprintf(tgtlayfn, "%s/target.lay", _dir);
  tgtlay = new Layout;
  tgtlay->load_file(tgtlayfn);

  assert(config["type"] == "judge");
  lowoff = (unsigned int)atoi(config["lowoff"].c_str());
  assert(lowoff < ctxlay->n);
  assert((ctxlay->n - lowoff) % 3 == 0);
  assert(tgtlay->n == 1);

  char dismapfn[4096], distopfn[4096];
  sprintf(distopfn, "%s/dis.top", _dir);
  sprintf(dismapfn, "%s/dis.map", _dir);
  distop = new Topology;
  distop->load_file(distopfn);
  dis = new Multitron(*distop, mbn, dismapfn);

  cumake(&cudistgt, dis->outn);
  cumake(&cudisin, dis->inn);
  cumake(&cudisfin, dis->inn);
  disfin = new double[dis->inn];

  ctxbuf = new double[mbn * ctxlay->n]();
  tgtbuf = new double[mbn * tgtlay->n]();

  assert(mbn * ctxlay->n == dis->inn);
  assert(mbn == dis->outn);

  rounds = 0;
}

Judge::~Judge() {
  delete ctxlay;
  delete tgtlay;

  cufree(cudisin);
  cufree(cudisfin);
  cufree(cudistgt);

  delete[] tgtbuf;
  delete[] ctxbuf;
  delete[] disfin;
}


void Judge::burn(double yo, double *modtgtbuf, bool update_stats) {
  assert(dis->inn == mbn * ctxlay->n);
  encude(ctxbuf, dis->inn, cudisin);

  assert(tgtlay->n == 1);
  assert(dis->outn == mbn);
  encude(tgtbuf, dis->outn, cudistgt);

  dis->feed(cudisin, modtgtbuf ? cudisfin : NULL);
  dis->target(cudistgt, update_stats);
  dis->train(yo);

  if (modtgtbuf) {
    double *tgtp = modtgtbuf;
    decude(cudisfin, dis->inn, disfin);
    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      for (unsigned int j = mbi * ctxlay->n + lowoff, jn = (mbi + 1) * ctxlay->n; j < jn; ++j) {
        *tgtp += disfin[j];
        ++tgtp;
      }
    }
  }
}

void Judge::save() {
  dis->sync(1);
}

void Judge::load() {
  dis->sync(0);
}

void Judge::report(const char *prog, FILE *outfp) {
  fprintf(
    outfp,
    "%s %s rounds=%u\n"
    "%s %s dis_err2=%g dis_errm=%g\n"
    "\n",
    prog, dir.c_str(), rounds,
    prog, dir.c_str(), dis->err2, dis->errm
  );
}

}

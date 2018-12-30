#include "project.hh"
#include "topology.hh"
#include "random.hh"
#include "cudamem.hh"

#include <math.h>

int main(int argc, char **argv) {
  assert(argc > 1);
  seedrand();

  unsigned int mbn = 8;
  ZoomProject *p = new ZoomProject(argv[1], mbn);
  unsigned int *mb = new unsigned int[mbn];

  unsigned int lfn = p->lofreqlay->n;
  unsigned int an = p->attrslay->n;
  unsigned int sn = p->sampleslay->n;
  unsigned int hfn = sn;
  unsigned int cn = p->contextlay->n;
  assert(cn == lfn + an);
  unsigned int csn = sn + cn;

  Dataset *hifreq = p->hifreq;
  Dataset *lofreq = p->lofreq;
  Dataset *attrs = p->attrs;

  assert(hifreq->n == lofreq->n);
  assert(hifreq->k == hfn);
  assert(lofreq->k == lfn);
  assert(attrs->k == an);
  assert(csn == hfn + lfn + an);
  assert(cn == lfn + an);

  Tron *encgentron = p->encgentron;

fprintf(stderr, "hfn=%u lfn=%u an=%u cn=%u sn=%u csn=%u mbn=%u\n", hfn, lfn, an, cn, sn, csn, mbn);
  double *encin = NULL;
  cumake(&encin, mbn * csn);
  const double *genout;

  double *gentgt = NULL;
  cumake(&gentgt, mbn * hfn);

assert(encgentron->inn == csn * mbn);
assert(encgentron->outn == hfn * mbn);

  unsigned int i = 0;

  while (1) {
    hifreq->pick_minibatch(mbn, mb);
    attrs->encude_minibatch(mb, mbn, encin, 0, csn);
    lofreq->encude_minibatch(mb, mbn, encin, an, csn);
    hifreq->encude_minibatch(mb, mbn, encin, cn, csn);

    genout = encgentron->feed(encin, NULL);
    hifreq->encude_minibatch(mb, mbn, gentgt);

    encgentron->target(gentgt);
    encgentron->train(0.5);

    if (i % 100 == 0) {
       fprintf(stderr, "i=%u cerr2=%lf cerrm=%lf\n", i, encgentron->cerr2, encgentron->cerrm);
       p->enctron->sync(1);
       p->gentron->sync(1);
    }
    ++i;
  }

  return 0;
}

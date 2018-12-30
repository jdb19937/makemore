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

  unsigned int nc = p->controlslay->n;
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
  Tron *gentron = p->gentron;

fprintf(stderr, "hfn=%u lfn=%u an=%u cn=%u sn=%u csn=%u mbn=%u\n", hfn, lfn, an, cn, sn, csn, mbn);
  double *encin = NULL;
  cumake(&encin, mbn * csn);
  const double *genout;

  double *genin;
  cumake(&genin, mbn * (nc + cn));
  double *gentgt = NULL;
  cumake(&gentgt, mbn * hfn);

assert(encgentron->inn == csn * mbn);
assert(encgentron->outn == hfn * mbn);

  unsigned int i = 0;
  double *r = new double[nc];

  while (1) {
    hifreq->pick_minibatch(mbn, mb);
    attrs->encude_minibatch(mb, mbn, encin, 0, csn);
    lofreq->encude_minibatch(mb, mbn, encin, an, csn);
    hifreq->encude_minibatch(mb, mbn, encin, cn, csn);

    genout = encgentron->feed(encin, NULL);
    hifreq->encude_minibatch(mb, mbn, gentgt);

    encgentron->target(gentgt);
    encgentron->train(0.02);



#if 0
    {
      hifreq->pick_minibatch(mbn, mb);
      attrs->encude_minibatch(mb, mbn, encin, 0, csn);
      lofreq->encude_minibatch(mb, mbn, encin, an, csn);
      hifreq->encude_minibatch(mb, mbn, encin, cn, csn);

      const double *encout = enctronpass->feed(encin);

    //encdistron->feed(encin, NULL);
    // distron->target(1)
    // distron->feed(rand
   // distron->target(0)
#endif


#if 0
    attrs->encude_minibatch(mb, mbn, genin, 0, nc+cn);
    lofreq->encude_minibatch(mb, mbn, genin, an, nc+cn);
    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      for (unsigned int j = 0; j < nc; ++j) { r[j] = 0.5; }
      encude(r, nc, genin + mbi * (nc + cn) + cn);
    }

    genout = gentron->feed(genin, NULL);
    gentron->target(gentgt);
    gentron->train(0.005);
#endif


  
    
     

    if (i % 100 == 0) {
       fprintf(stderr, "i=%u cerr2=%lf cerrm=%lf\n", i, encgentron->cerr2, encgentron->cerrm);
       p->enctron->sync(1);
       p->gentron->sync(1);
    }
    ++i;
  }

  return 0;
}

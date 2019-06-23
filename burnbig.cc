#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>

#include "urb.hh"
#include "zone.hh"
#include "cudamem.hh"
#include "numutils.hh"
#include "imgutils.hh"
#include "encgen.hh"
#include "cholo.hh"
#include "strutils.hh"
#include "partrait.hh"
#include "catalog.hh"


using namespace makemore;
using namespace std;

int main() {
  seedrand();

  unsigned int mbn = 1;
  Encgen egd("bignew.proj", mbn);
  double *tmpd = new double[1<<20];
  unsigned int w = 256, h = 256;

  Zone zone("/spin/dan/shampane.dat");
//  Catalog cat("/spin/dan/celeba.aligned");

  assert(egd.tgtlay->n == w * h * 3);

fprintf(stderr, "starting\n");

Partrait *par = new Partrait;
//  Partrait *spar = new Partrait[1024];

  int i = 0;
  while (1) {
#if 1
    Parson *prs = zone.pick();
    assert(prs);

    std::string fn = prs->srcfn;
    assert(fn.length());

    par->load(fn);
#else
    cat.pick(par);
#endif

    bool reflected = (randuint() % 2);
reflected = 0;
    if (reflected)
      par->reflect();

    assert(par->w * par->h * 3 == egd.tgtlay->n);
    rgblab(par->rgb, egd.tgtlay->n, egd.tgtbuf);

    assert(egd.ctxlay->n >= 192 + 64 + 3);
    par->make_sketch(egd.ctxbuf);

    Hashbag hb;
    bool no_tags = (randuint() % 128 == 0);
no_tags = 0;
    if (!no_tags)
      par->bag_tags(&hb);

    bool rand_tag = (randuint() % 128 == 0);
rand_tag = 0;
    if (rand_tag) {
      char buf[256];
      sprintf(buf, "hello%u", randuint());
      hb.add(buf);
    }

    assert(Hashbag::n >= 64);
    memcpy(egd.ctxbuf + 192, hb.vec, sizeof(double) * 64);

    memset(egd.ctxbuf + 256, 0, sizeof(double) * 16);
    egd.ctxbuf[256] = par->get_tag("angle", 0.0);
    egd.ctxbuf[257] = par->get_tag("stretch", 1.0);
    egd.ctxbuf[258] = par->get_tag("skew", 0.0);

    egd.burn(0.00000, 0.00001);

#if 1
    if (!no_tags && !rand_tag && !reflected) {
      assert(Parson::ncontrols == egd.ctrlay->n);
      decude(egd.enc->output(), Parson::ncontrols, prs->controls);
      prs->revised = time(NULL);

      prs->recon_err = sqrt(cusumsq(egd.gen->foutput(), egd.tgtlay->n) / (double)egd.tgtlay->n);
    }
#endif

    ++i;
    if (i % 100 == 0) {
      egd.report("burnbig");
fprintf(stderr, "saving\n");
      egd.save();
fprintf(stderr, "saved\n");
    }
  }
}


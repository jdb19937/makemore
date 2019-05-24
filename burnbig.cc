#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>

#include "urb.hh"
#include "zone.hh"
#include "cudamem.hh"
#include "numutils.hh"
#include "imgutils.hh"
#include "encgendis.hh"
#include "cholo.hh"
#include "strutils.hh"
#include "partrait.hh"
#include "catalog.hh"


using namespace makemore;
using namespace std;

int main() {
  seedrand();

  unsigned int mbn = 1;
  Encgendis egd("big.proj", mbn);
  double *tmpd = new double[1<<20];
  unsigned int w = 256, h = 256;

  Catalog cat;
  cat.add_dir("/spin/dan/celeba.aligned");
  cat.add_dir("/spin/dan/shampane.aligned");
  cat.add_dir("/spin/dan/dancam.aligned");

  assert(egd.tgtlay->n == w * h * 3);

fprintf(stderr, "starting\n");

Partrait *par = new Partrait;
//  Partrait *spar = new Partrait[1024];

  int i = 0;
  while (1) {
//    if (i % 1024 == 0) {
//      fprintf(stderr, "loading new samples\n");
//      cat.pick(spar, 1024);
//    }

//    Partrait *par = spar + (i % 1024);

    cat.pick(par, 1, true);

    assert(par->w * par->h * 3 == egd.tgtlay->n);
    rgblab(par->rgb, egd.tgtlay->n, egd.tgtbuf);

    assert(egd.ctxlay->n >= 192 + 64 + 3);
    par->make_sketch(egd.ctxbuf);

    Hashbag hb;
    if (randuint() % 128)
      par->bag_tags(&hb);
    assert(Hashbag::n >= 64);
    memcpy(egd.ctxbuf + 192, hb.vec, sizeof(double) * 64);

    memset(egd.ctxbuf + 256, 0, sizeof(double) * 16);
    egd.ctxbuf[256] = par->get_tag("angle", 0.0);
    egd.ctxbuf[257] = par->get_tag("stretch", 1.0);
    egd.ctxbuf[258] = par->get_tag("skew", 0.0);

    egd.burn(0.0002, 0.0002);

    if (i % 100 == 0) {
      egd.report("burnbig");
fprintf(stderr, "saving\n");
      egd.save();
fprintf(stderr, "saved\n");
    }
    ++i;
  }
}


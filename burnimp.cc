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
#include "impdis.hh"


using namespace makemore;
using namespace std;

int main() {
  seedrand();

  unsigned int mbn = 1;
  Encgen egd("bignew.proj", mbn);
  Impdis id("id.proj", mbn);
  unsigned int w = 256, h = 256;

  Catalog cat;
  cat.add_dir("/spin/dan/celeba.aligned");

  fprintf(stderr, "starting\n");

  Partrait *par = new Partrait;
  int i = 0;
  while (1) {
    cat.pick(par, 1, true);

    assert(par->w * par->h * 3 == egd.tgtlay->n);
    rgblab(par->rgb, id.tgtlay->n, id.tgtbuf);

    memcpy(egd.tgtbuf, id.tgtbuf, sizeof(double) * egd.tgtlay->n);
    egd.encode();

    assert(egd.ctxlay->n >= 192 + 64 + 3);
    par->make_sketch(egd.ctxbuf);

    Hashbag hb;
    assert(Hashbag::n >= 64);
    memcpy(egd.ctxbuf + 192, hb.vec, sizeof(double) * 64);

    memset(egd.ctxbuf + 256, 0, sizeof(double) * 16);
    egd.ctxbuf[256] = par->get_tag("angle", 0.0);
    egd.ctxbuf[257] = par->get_tag("stretch", 1.0);
    egd.ctxbuf[258] = par->get_tag("skew", 0.0);

    egd.generate();
    memcpy(id.inbuf, egd.tgtbuf, sizeof(double) * egd.tgtlay->n);

    id.burn(0.00005);
//    id.observe(0.0005);

    ++i;
    if (i % 100 == 0) {
      egd.load();
      id.report("burnimp");
fprintf(stderr, "saving\n");
      id.save();
fprintf(stderr, "saved\n");
    }
  }
}


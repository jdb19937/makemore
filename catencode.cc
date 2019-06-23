#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>

#include "urb.hh"
#include "zone.hh"
#include "cudamem.hh"
#include "numutils.hh"
#include "imgutils.hh"
#include "encoder.hh"
#include "cholo.hh"
#include "strutils.hh"
#include "partrait.hh"
#include "catalog.hh"
#include "generator.hh"


using namespace makemore;
using namespace std;

int main(int argc, char **argv) {
  seedrand();

  unsigned int mbn = 1;
  Encoder enc("newenc.proj", mbn);
//  Generator gen("gen.shampane.proj", mbn);
  unsigned int w = 256, h = 256;

  assert(argc > 1);
  Zone zone(argv[1]);

//  assert(gen.tgtlay->n == w * h * 3);
  assert(enc.tgtlay->n == w * h * 3);

fprintf(stderr, "starting\n");

Partrait *par = new Partrait;
//  Partrait *spar = new Partrait[1024];

  int i = 0;
  while (i < zone.n) {
    Parson *prs = zone.db + i;
    std::string fn = prs->srcfn;
    assert(fn.length());

    Partrait prt;
    prt.load(fn);

    enc.encode(prt, prs);

    prs->revised = time(NULL);

//    Partrait tmp;
//    gen.generate(*prs, &tmp);

//    gen.gen->target(enc.cuencin);

//    assert(gen.focus);
//    cufocus(gen.gen->foutput(), gen.cutgtlayx, gen.cutgtlayy, gen.tgtlay->n);
//    prs->recon_err = sqrt(cusumsq(gen.gen->foutput(), gen.tgtlay->n) / (double)gen.tgtlay->n);
//fprintf(stdout, "%s\t%lf\t%lf\t%lf\n", prs->srcfn, prs->recon_err, prs->stretch, prs->skew);
//fflush(stdout);

    ++i;
  }
}


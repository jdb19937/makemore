#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>

#include "urb.hh"
#include "zone.hh"
#include "cudamem.hh"
#include "numutils.hh"
#include "imgutils.hh"
#include "superenc.hh"
#include "cholo.hh"
#include "strutils.hh"
#include "partrait.hh"
#include "catalog.hh"
#include "supergen.hh"
#include "tmutils.hh"


using namespace makemore;
using namespace std;

void maketri(Partrait *prt) {
  Point p(randrange(0, 100), randrange(155, 255));
  Point q(randrange(100, 155), randrange(0, 100));
  Point r(randrange(155, 255), randrange(155, 255));

  Triangle tri(p, q, r);
  uint8_t *rgb = prt->rgb;
  for (unsigned int y = 0; y < 256; ++y) {
    for (unsigned int x = 0; x < 256; ++x) {
      Point a(x, y);
      if (tri.contains(a)) {
        *rgb++ = 255;
        *rgb++ = 255;
        *rgb++ = 255;
      } else {
        *rgb++ = 0;
        *rgb++ = 0;
        *rgb++ = 0;
      }
    }
  }
}


int main(int argc, char **argv) {
  seedrand();

  unsigned int mbn = 1;
  assert(argc > 2);
  Superenc enc(argv[1], mbn);
  std::vector<Supergen*> gens;
  for (unsigned int i = 2; i < argc; ++i)
    gens.push_back(new Supergen(argv[i], mbn));
  unsigned int w = 256, h = 256;

  assert(enc.inplay->n == w * h * 3);

fprintf(stderr, "starting\n");

Partrait *par = new Partrait;
//  Partrait *spar = new Partrait[1024];

  double *ctr = new double[enc.enc->outn];
double t0 = now();

//Cholo *cholo = new Cholo(enc.ctrlay->n);

  int i = 0;
  while (1) {
    Supergen *genp = gens[randuint() % gens.size()];
    Supergen &gen(*genp);

    Zone *zonep = gen.zone;
    assert(zonep);
    Zone &zone(*zonep);

#if 1
    Parson *prs;
//do {
    prs = zone.pick();
//} while (prs->has_tag("glasses"));
    std::string fn = prs->srcfn;
    assert(fn.length());

    Partrait prt;
//fprintf(stderr, "loading partrait %s\n", fn.c_str());
    prt.load(fn);
    if (randuint() % 2)
      prt.reflect();

#else
    Partrait prt(256, 256);
    maketri(&prt);
#endif

//fprintf(stderr, " loaded partrait %s\n", fn.c_str());


    // prt.jitter();


    enc.encode(prt, ctr);
    gen.generate(ctr, NULL, NULL, true);

    gen.burn(prt, 0.0003);
    // enc.burn(gen, 0.00003);

//cholo->observe(ctr);

    ++i;

   if (i % 100 == 0) {

    gen.generate(ctr, &prt);
prt.save("/var/www/html/test.png");

if (i % 1000 == 0) {
  //cholo->finalize();

  for (unsigned int j = 0; j < enc.ctrlay->n; ++j)
    ctr[j] = randgauss();
  //cholo->generate(ctr, ctr);
  gen.generate(ctr, &prt);
prt.save("/var/www/html/testr.png");

//  delete cholo;
//  cholo = new Cholo(enc.ctrlay->n);
}

      enc.report("burnsuper");
      for (auto g : gens)
        g->report("burnsuper");

      fprintf(stderr, "saving\n");
      for (auto g : gens)
        g->save();
      enc.save();
      fprintf(stderr, "saved\n");
double t1 = now();
double dt = t1 - t0;
fprintf(stderr, "dt=%lf\n", t1 - t0);

t0 = now();

    }
  }
}


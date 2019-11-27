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
#include "superenc.hh"
#include "video.hh"
#include "superdis.hh"


using namespace makemore;
using namespace std;

void maketri(Partrait *prt) {
prt->create(256, 256);
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

Parson *pk(Zone &zone) {
  Parson *prs;
//  do {
    prs = zone.pick();
//  } while (prs->has_tag("glasses"));
  return prs;
}

int main(int argc, char **argv) {
  seedrand();

  unsigned int mbn = 1;
  assert(argc > 3);
  Superenc enc(argv[1], 1);
  Supergen gen(argv[2], 1);
//  Superenc enc0("/spin/dan/nbak/nenc.proj", 1);
//  Supergen gen0("/spin/dan/nbak/ngen.proj", 1);
  Superenc enc0("penc.proj", 1);
  Supergen gen0("pgen.proj", 1);
  Superdis dis(argv[3], 1);
  unsigned int w = 256, h = 256;

  assert(enc.inplay->n == w * h * 3);

Parson *prsv = pk(*gen.zone);

fprintf(stderr, "starting\n");

  double *ctr = new double[enc.enc->outn];
double t0 = now();

  int i = 0;
  while (1) {
    Zone *zonep = gen.zone;
    assert(zonep);
    Zone &zone(*zonep);

    Partrait prt2;
    std::string fn0 = pk(zone)->srcfn;
    std::string fn1 = pk(zone)->srcfn;
    std::string fn2 = pk(zone)->srcfn;
//fprintf(stderr, "fn0=%s fn1=%s fn2=%s\n", fn0.c_str(), fn1.c_str(), fn2.c_str());

    for (unsigned int r = 0; r < 1; ++r) {
#if 1
    Partrait prt0, prt1;
#if 1
    prt0.load(fn0);
    prt1.load(fn1);
    if (randuint() % 2)
      prt0.reflect();
    if (randuint() % 2)
      prt1.reflect();
#else
  maketri(&prt0);
  maketri(&prt1);
#endif
#endif

    dis.observe(&prt0, &enc, &gen, &prt1, 1e-4);
    }


    for (unsigned int r = 0; r < 1; ++r) {
#if 1

#if 1
#if 1
    prt2.load(fn2);
    if (randuint() % 2)
      prt2.reflect();
#else
    maketri(&prt2);
#endif
#endif

    enc.encode(prt2, ctr);
//if (i > 100) {
    gen.generate(ctr, NULL, NULL, true);

    gen.burn(prt2, 1e-4, &dis);
    // enc.burn(gen, 1e-4);


//}

#endif
    }

   if (i % 100 == 0) {

//prt2.load(prsv->srcfn);
    enc.encode(prt2, ctr);
//for (unsigned int j = 0; j < enc.ctrlay->n; ++j)
//  ctr[j] = (int8_t)(ctr[j] * 127.0) / 127.0;

Partrait prt3(256 ,256);
gen.generate(ctr, &prt3);
Partrait prt4(256, 256);
enc0.encode(prt2, ctr);
gen0.generate(ctr, &prt4);
Partrait prt5(256, 1024+256);
memcpy(prt5.rgb, prt3.rgb, 256*256*3);
memcpy(prt5.rgb+256*256*3, prt4.rgb, 256*256*3);
memcpy(prt5.rgb+256*256*3*3, prt2.rgb, 256*256*3);
uint8_t *p = prt5.rgb+256*256*3*2;
double m = 0.5;
for (unsigned int j = 0;j < 256*256*3; ++j) {
  p[j] = 128 + m * ((int)prt3.rgb[j] - (int)prt4.rgb[j]);
}
p = prt5.rgb+256*256*3*4;
for (unsigned int j = 0;j < 256*256*3; ++j) {
  p[j] = 128 + m * ((int)prt2.rgb[j] - (int)prt3.rgb[j]);
}
prt5.save("/var/www/html/foo2.png");

      enc.report("burndis");
      dis.report("burndis");
      gen.report("burndis");
      fprintf(stderr, "saving\n");
      dis.save();
gen.save();
enc.save();
      fprintf(stderr, "saved\n");

double t1 = now();
double dt = t1 - t0;
fprintf(stderr, "dt=%lf\n", t1 - t0);
t0 = now();

    }
    ++i;
  }
}


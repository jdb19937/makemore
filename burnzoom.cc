#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>

#include "urb.hh"
#include "zone.hh"
#include "cudamem.hh"
#include "numutils.hh"
#include "imgutils.hh"
#include "cholo.hh"
#include "strutils.hh"
#include "partrait.hh"
#include "catalog.hh"
#include "zoomgen.hh"
#include "zoomdis.hh"
#include "tmutils.hh"


using namespace makemore;
using namespace std;

int main(int argc, char **argv) {
double *ctr = new double[1024];
  seedrand();

assert(argc == 4 || argc == 5);
  Catalog cat(argv[3]);
  Catalog discat(argc > 4 ? argv[4] : argv[3]);

  Zoomgen gen(argv[1], 1);
  Zoomdis dis(argv[2], 1);
  unsigned int wout = 32, hout = 32;

  double t0 = now();

  int i = 0;
  while (1) {
    if (1) {
      Partrait prt0, prt1;
      cat.pick(&prt0, 1, 0);
      discat.pick(&prt1, 1, 0);

      unsigned int x0 = randuint() % (prt0.w - 32);
      unsigned int y0 = randuint() % (prt0.h - 32);
      Partrait p0(32, 32);
      prt0.cut(&p0, x0, y0);

      unsigned int x1 = randuint() % (prt1.w - 32);
      unsigned int y1 = randuint() % (prt1.h - 32);
      Partrait p1(32, 32);
      prt1.cut(&p1, x1, y1);

      // p0.shrink();

      dis.observe(&p0, &gen, &p1, 1e-4);
    }


    {

    Partrait prt;
    cat.pick(&prt, 1, 0);

    unsigned int x = randuint() % (prt.w - 32);
    unsigned int y = randuint() % (prt.h - 32);
    Partrait pout(32, 32);
    prt.cut(&pout, x, y);

    Partrait pin(pout);
    assert(pin.w == 32 && pin.h == 32);
    // pin.shrink();
    
    gen.generate(pin);
    gen.burn(0.0001, &dis, &pout, 0);

   if (i % 100 == 0) {
pout.save("./zoomsrc.png");

    gen.generate(pin, &pout);
    pout.save("./zoom.png");

    gen.generate(pin, &pout);
    pout.save("./zoom2.png");

      gen.report("burnzoom");
      dis.report("burnzoom");

      fprintf(stderr, "saving\n");
      gen.save();
      dis.save();
      fprintf(stderr, "saved\n");
double t1 = now();
double dt = t1 - t0;
fprintf(stderr, "dt=%lf\n", t1 - t0);

t0 = now();

    }

    ++i;
    }
  }
}


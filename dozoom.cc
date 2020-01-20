#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>

#include "urb.hh"
#include "cudamem.hh"
#include "numutils.hh"
#include "imgutils.hh"
#include "cholo.hh"
#include "strutils.hh"
#include "partrait.hh"
#include "catalog.hh"
#include "zoomgen.hh"
#include "tmutils.hh"


using namespace makemore;
using namespace std;

int main(int argc, char **argv) {
  seedrand();

  unsigned int mbn = 1;
assert(argc > 1);
  Zoomgen gen(argv[1], 1);

Partrait *par = new Partrait;
//  Partrait *spar = new Partrait[1024];

double t0 = now();

//Cholo *cholo = new Cholo(enc.ctrlay->n);

  double *buf = new double[1024 * 1024 * 3];

Partrait prt;
prt.load(stdin);
Partrait prtout;
gen.generate(prt, &prtout);
prtout.save(stdout);
return 0;
}


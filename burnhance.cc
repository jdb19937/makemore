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
#include "enhancer.hh"


using namespace makemore;
using namespace std;

int main(int argc, char **argv) {
  seedrand();

  Enhancer enh(argv[1]);

  std::string sampfn = argv[2];
  FILE *sampfp = fopen(sampfn.c_str(), "r");
  assert(sampfp);

  fseek(sampfp, 0, 2);
  long samptop = ftell(sampfp);
  assert(samptop > 0);
  assert(samptop % (32 * 32 * 9) == 0);
  long sampn = samptop / (32 * 32 * 9);

  uint8_t *sampbuf = new uint8_t[32 * 32 * 9];
  double *dsamp = new double[32 * 32 * 9];
  double *cudsamp;
  cumake(&cudsamp, 32 * 32 * 9);

  double t0 = now();

  int ret;
  int i = 0;

  while (1) {
    int sampi = randuint() % sampn;

    ret = fseek(sampfp, sampi * 32 * 32 * 9, 0);
    assert(ret == 0);
    ret = fread(sampbuf, 1, 32 * 32 * 9, sampfp);
    assert(ret == 32 * 32 * 9);

    btodv(sampbuf, dsamp, 32 * 32 * 9);
    encude(dsamp, 32 * 32 * 9, cudsamp);

    enh.observe(cudsamp, 1e-4);

    if (i % 100 == 0) {
     enh.report("burnhance");

     fprintf(stderr, "saving\n");
     enh.save();
     fprintf(stderr, "saved\n");

double t1 = now();
double dt = t1 - t0;
fprintf(stderr, "dt=%lf\n", t1 - t0);
t0 = now();
    }

    ++i;
  }
}


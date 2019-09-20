#define __MAKEMORE_SUPERENC_CC__ 1

#include <string>
#include <algorithm>

#include "cudamem.hh"
#include "tron.hh"
#include "multitron.hh"
#include "superenc.hh"
#include "parson.hh"
#include "strutils.hh"
#include "imgutils.hh"
#include "supergen.hh"

namespace makemore {

using namespace std;

Superenc::Superenc(const std::string &_dir, unsigned int _mbn) : Project(_dir, _mbn) {
  assert(mbn > 0);

  assert(config["type"] == "superenc");

  char ctrlayfn[4096];
  sprintf(ctrlayfn, "%s/control.lay", dir.c_str());
  ctrlay = new Layout;
  ctrlay->load_file(ctrlayfn);

  char inplayfn[4096];
  sprintf(inplayfn, "%s/input.lay", dir.c_str());
  inplay = new Layout;
  inplay->load_file(inplayfn);

  char encmapfn[4096];
  sprintf(encmapfn, "%s/enc.map", dir.c_str());
  encmap = new Mapfile(encmapfn);
  enc = new Supertron(encmap);

  encinlay = new Layout(*inplay);
  assert(enc->inn == mbn * encinlay->n);
  assert(enc->outn == mbn * ctrlay->n);

  cumake(&cuencin, enc->inn);

  ctrbuf = new double[mbn * ctrlay->n]();
  inpbuf = new double[mbn * inplay->n]();

  rounds = 0;
}

Superenc::~Superenc() {
  delete encinlay;

  delete inplay;
  delete ctrlay;

  cufree(cuencin);

  delete[] ctrbuf;
  delete[] inpbuf;
}


void Superenc::report(const char *prog) {
  fprintf(
    stderr,
    "%s %s rounds=%u enc_err2=%g enc_errm=%g\n",
    prog, dir.c_str(), rounds, enc->err2, enc->errm
  );
}

void Superenc::save() {
  encmap->save();
}

void Superenc::load() {
  encmap->load();
}

void Superenc::encode(const Partrait &prt, double *ctr) {
  assert(mbn == 1);
  assert(encinlay->n == inplay->n);
  assert(inplay->n == prt.w * prt.h * 3);

  rgblab(prt.rgb, inplay->n, inpbuf);
  encude( inpbuf, inplay->n, cuencin);

  const double *cuencout = enc->feed(cuencin, NULL);
  assert(enc->outn == ctrlay->n);
  decude(cuencout, enc->outn, ctr);
}

void Superenc::burn(const class Supergen &gen, double nu) {
  assert(gen.ctrlay->n == ctrlay->n);
if (0) {
double foo[1024];
decude(gen.cugenfin, 1024, foo);
for (unsigned int j = 0; j < 16; ++j)
fprintf(stderr,"%lf\t", foo[j]);
fprintf(stderr,"\n");
}
  cucopy(gen.cugenfin + gen.ctxlay->n, gen.ctrlay->n, enc->foutput());
  enc->update_stats();
  enc->train(nu);
}




#if 0
  assert(geninlay->n == ctxlay->n + ctrlay->n);
  assert(encinlay->n == inplay->n);
  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    encude(ctxbuf + mbi * ctxlay->n, ctxlay->n, cugenin + mbi * geninlay->n);
    encude(inpbuf + mbi * inplay->n, inplay->n, cuencin + mbi * encinlay->n);
  }

  const double *cuencout = enc->feed(cuencin, NULL);
  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    cucopy(cuencout + mbi * ctrlay->n, ctrlay->n, cugenin + mbi * geninlay->n + ctxlay->n);
  }

  cuzero(cugenfin, mbn * geninlay->n);
  gen->feed(cugenin, cugenfin);

  assert(gen->outn == mbn * inplay->n);
  encude(inpbuf, gen->outn, cugeninp);
  gen->target(cugeninp, false);

  if (focus) {
    double *cugenfout = gen->foutput();
    for (unsigned int mbi = 0; mbi < mbn; ++mbi)
      cufocus(cugenfout + mbi * inplay->n, cuinplayx, cuinplayy, inplay->n);
  }

  gen->update_stats();
  gen->train(pi);

  if (nu > 0) {
    double *cuencfoutput = enc->foutput();
    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      cucopy(cugenfin + mbi * geninlay->n + ctxlay->n, ctrlay->n, cuencfoutput + mbi * ctrlay->n);
    }

    enc->update_stats();
    enc->train(nu);
  }
#endif

}

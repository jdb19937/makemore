#define __MAKEMORE_SUPRGEN_CC__ 1

#include <string>
#include <algorithm>

#include "cudamem.hh"
#include "tron.hh"
#include "multitron.hh"
#include "encoder.hh"
#include "sndgen.hh"
#include "parson.hh"
#include "strutils.hh"
#include "imgutils.hh"
#include "cholo.hh"
#include "numutils.hh"

namespace makemore {

using namespace std;

Sndgen::Sndgen(const std::string &_dir, unsigned int _mbn) : Project(_dir, _mbn) {
  assert(mbn > 0);

  assert(config["type"] == "sndgen");

  char ctrlayfn[4096];
  sprintf(ctrlayfn, "%s/control.lay", dir.c_str());
  ctrlay = new Layout;
  ctrlay->load_file(ctrlayfn);

  char tgtlayfn[4096];
  sprintf(tgtlayfn, "%s/target.lay", dir.c_str());
  tgtlay = new Layout;
  tgtlay->load_file(tgtlayfn);

  char genmapfn[4096];
  sprintf(genmapfn, "%s/gen.map", dir.c_str());
  genmap = new Mapfile(genmapfn);
  gen = new Supertron(genmap);

  geninlay = new Layout(*ctrlay);
  assert(gen->inn == mbn * geninlay->n);
  assert(gen->outn == mbn * tgtlay->n);

  cumake(&cugentgt, gen->outn);
  cumake(&cugenin, gen->inn);
  cumake(&cugenfin, gen->inn);

  ctrbuf = new double[mbn * ctrlay->n]();
  tgtbuf = new double[mbn * tgtlay->n]();
  buf = new double[mbn * tgtlay->n]();

  rounds = 0;
}

Sndgen::~Sndgen() {
  delete geninlay;

  delete tgtlay;
  delete ctrlay;

  cufree(cugenin);
  cufree(cugenfin);

  delete[] ctrbuf;
  delete[] tgtbuf;
  delete[] buf;
}


void Sndgen::report(const char *prog) {
  fprintf(
    stderr,
    "%s %s rounds=%u gen_err2=%g gen_errm=%g\n",
    prog, dir.c_str(), rounds, gen->err2, gen->errm
  );
}

void Sndgen::save() {
  genmap->save();
}

void Sndgen::load() {
  genmap->load();
}

void Sndgen::generate(const double *ctr, Soundpic *sndpic, bool bp) {
  assert(mbn == 1);

  encude(ctr, ctrlay->n, cugenin);

  const double *cugenout;
  if (bp) {
    cuzero(cugenfin, gen->inn);
    cugenout = gen->feed(cugenin, cugenfin);
  } else {
    cugenout = gen->feed(cugenin, NULL);
  }

  if (sndpic) {
    assert(sndpic->w * sndpic->h == gen->outn);
    decude(cugenout, gen->outn, sndpic->tab);
  }
}

void Sndgen::burn(const Soundpic &sndpic, double pi) {
  assert(sndpic.w * sndpic.h == gen->outn);
  assert(sndpic.w * sndpic.h == tgtlay->n);
  encude(sndpic.tab, tgtlay->n, cugentgt);
  gen->target(cugentgt);

  gen->update_stats();
  gen->train(pi);
}

}

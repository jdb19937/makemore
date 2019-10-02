#define __MAKEMORE_SNDENC_CC__ 1

#include <string>
#include <algorithm>

#include "cudamem.hh"
#include "tron.hh"
#include "multitron.hh"
#include "sndenc.hh"
#include "parson.hh"
#include "strutils.hh"
#include "imgutils.hh"
#include "sndgen.hh"
#include "soundpic.hh"

namespace makemore {

using namespace std;

Sndenc::Sndenc(const std::string &_dir, unsigned int _mbn) : Project(_dir, _mbn) {
  assert(mbn > 0);

  assert(config["type"] == "sndenc");

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

Sndenc::~Sndenc() {
  delete encinlay;

  delete inplay;
  delete ctrlay;

  cufree(cuencin);

  delete[] ctrbuf;
  delete[] inpbuf;
}


void Sndenc::report(const char *prog) {
  fprintf(
    stderr,
    "%s %s rounds=%u enc_err2=%g enc_errm=%g\n",
    prog, dir.c_str(), rounds, enc->err2, enc->errm
  );
}

void Sndenc::save() {
  encmap->save();
}

void Sndenc::load() {
  encmap->load();
}

void Sndenc::encode(const Soundpic &sndpic, double *ctr) {
  assert(mbn == 1);
  assert(encinlay->n == inplay->n);
  assert(inplay->n == sndpic.w * sndpic.h);

  encude(sndpic.tab, inplay->n, cuencin);

  const double *cuencout = enc->feed(cuencin, NULL);
  assert(enc->outn == ctrlay->n);
  decude(cuencout, enc->outn, ctr);
}

void Sndenc::burn(const class Sndgen &gen, double nu) {
  assert(gen.ctrlay->n == ctrlay->n);
  cucopy(gen.cugenfin, gen.ctrlay->n, enc->foutput());
  enc->update_stats();
  enc->train(nu);
}


}

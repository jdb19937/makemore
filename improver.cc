#define __MAKEMORE_IMPROVER_CC__ 1

#include <string>
#include <netinet/in.h>

#include "improver.hh"
#include "project.hh"
#include "cudamem.hh"
#include "tron.hh"
#include "multitron.hh"
#include "twiddle.hh"
#include "closest.hh"

namespace makemore {

using namespace std;

Improver::Improver(const char *_dir, unsigned int _mbn) : Project(_dir, _mbn) {
  char ctxlayfn[4096];
  sprintf(ctxlayfn, "%s/context.lay", _dir);
  ctxlay = new Layout;
  ctxlay->load_file(ctxlayfn);

  char ctrlayfn[4096];
  sprintf(ctrlayfn, "%s/control.lay", _dir);
  ctrlay = new Layout;
  ctrlay->load_file(ctrlayfn);

  char tgtlayfn[4096];
  sprintf(tgtlayfn, "%s/target.lay", _dir);
  tgtlay = new Layout;
  tgtlay->load_file(tgtlayfn);

  assert(config["type"] == "improver");

  assert(ctxlay->n >= tgtlay->n);


  char encmapfn[4096], enctopfn[4096];
  sprintf(enctopfn, "%s/enc.top", _dir);
  sprintf(encmapfn, "%s/enc.map", _dir);
  enctop = new Topology;
  enctop->load_file(enctopfn);
  enc = new Multitron(*enctop, mbn, encmapfn);

  char genmapfn[4096], gentopfn[4096];
  sprintf(gentopfn, "%s/gen.top", _dir);
  sprintf(genmapfn, "%s/gen.map", _dir);
  gentop = new Topology;
  gentop->load_file(gentopfn);
  gen = new Multitron(*gentop, mbn, genmapfn);

  if (config["activated"] == "1") {
    gen->mt1->activated = true;
  } else {
    assert(config["activated"] == "0" || config["activated"] == "");
    gen->mt1->activated = false;
  }

  assert(tgtlay->n + 64 == ctxlay->n);

  encinlay = new Layout(*ctxlay);
  *encinlay += *tgtlay;
  assert(enc->inn == mbn * encinlay->n);
  assert(enc->outn == mbn * ctrlay->n);

  geninlay = new Layout(*ctxlay);
  *geninlay += *ctrlay;
  assert(gen->inn == mbn * geninlay->n);
  assert(gen->outn == mbn * tgtlay->n);


  encpass = new Passthrutron(ctxlay->n, mbn, enc);
  genpass = new Passthrutron(ctxlay->n, mbn, gen);

  encgen = new Compositron(encpass, gen);
  genenc = new Compositron(genpass, enc);

  realctr = new double[mbn * ctrlay->n];
  fakectr = new double[mbn * ctrlay->n];
  fakectx = new double[mbn * ctxlay->n];
  morectr = new double[mbn * ctrlay->n];

  cumake(&cuenctgt, enc->outn);
  cumake(&cuencin, enc->inn);
  cumake(&cugentgt, gen->outn);
  cumake(&cugenin, gen->inn);

  bctxbuf = new uint8_t[mbn * ctxlay->n]();
  bctrbuf = new uint8_t[mbn * ctrlay->n]();
  btgtbuf = new uint8_t[mbn * tgtlay->n]();
  boutbuf = new uint8_t[mbn * tgtlay->n]();
  sadjbuf = new uint16_t[mbn * tgtlay->n]();
  bsepbuf = new uint8_t[mbn * (ctxlay->n + tgtlay->n)];

  ctxbuf = new double[mbn * ctxlay->n]();
  ctrbuf = new double[mbn * ctrlay->n]();
  tgtbuf = new double[mbn * tgtlay->n]();
  outbuf = new double[mbn * tgtlay->n]();
  adjbuf = new double[mbn * tgtlay->n]();
  sepbuf = new double[mbn * (ctxlay->n + tgtlay->n)];

  cumake(&cutgtlayx, tgtlay->n);
  encude(tgtlay->x, tgtlay->n, cutgtlayx);

  cumake(&cutgtlayy, tgtlay->n);
  encude(tgtlay->y, tgtlay->n, cutgtlayy);

  rounds = 0;
}

Improver::~Improver() {
  delete encpass; 
  delete genpass;
  delete encgen;
  delete genenc;
  delete encinlay;
  delete geninlay;

  delete ctxlay;
  delete tgtlay;
  delete ctrlay;

  cufree(cugenin);
  cufree(cuencin);
  cufree(cugentgt);
  cufree(cuenctgt);

  cufree(cutgtlayx);
  cufree(cutgtlayy);

  delete[] realctr;
  delete[] fakectr;
  delete[] fakectx;
  delete[] morectr;
  delete[] ctxbuf;
  delete[] tgtbuf;
  delete[] sepbuf;
  delete[] outbuf;
  delete[] adjbuf;

  delete[] bctxbuf;
  delete[] btgtbuf;
  delete[] boutbuf;
  delete[] bsepbuf;
  delete[] sadjbuf;
}


void Improver::burn(double nu, double pi) {
  assert(encinlay->n == ctxlay->n + tgtlay->n);
  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    encude(ctxbuf + mbi * ctxlay->n, ctxlay->n, cuencin + mbi * encinlay->n + 0);
    encude(tgtbuf + mbi * tgtlay->n, tgtlay->n, cuencin + mbi * encinlay->n + ctxlay->n);
  }
  assert(gen->outn == mbn * tgtlay->n);
  encude(tgtbuf, gen->outn, cugentgt);

  encgen->feed(cuencin, NULL);
  encgen->target(cugentgt);

#if 0
  double *cugenfout = gen->foutput();
  for (unsigned int mbi = 0; mbi < mbn; ++mbi)
    cufocus(cugenfout + mbi * tgtlay->n, cutgtlayx, cutgtlayy, tgtlay->n);
#endif

  gen->train(pi);

  encpass->update_stats();
  encpass->train(nu);
}

void Improver::report(const char *prog, FILE *outfp) {
  fprintf(
    outfp,
    "%s %s rounds=%u\n"
    "%s %s encgen_err2=%g encgen_errm=%g\n"
    "%s %s encpass_err2=%g encpass_errm=%g\n"
    "%s %s enc_err2=%g enc_errm=%g\n"
    "%s %s gen_err2=%g gen_errm=%g\n"
    "%s %s genpass_err2=%g genpass_errm=%g\n"
    "%s %s genenc_err2=%g genenc_errm=%g\n"
    "\n",
    prog, dir.c_str(), rounds,
    prog, dir.c_str(), encgen->err2, encgen->errm,
    prog, dir.c_str(), encpass->err2, encpass->errm,
    prog, dir.c_str(), enc->err2, enc->errm,
    prog, dir.c_str(), gen->err2, gen->errm,
    prog, dir.c_str(), genpass->err2, genpass->errm,
    prog, dir.c_str(), genenc->err2, genenc->errm
  );
}

void Improver::save() {
  enc->sync(1);
  gen->sync(1);
}

void Improver::load() {
  enc->sync(0);
  gen->sync(0);
}

void Improver::scramble(double mean, double dev) {
  for (unsigned int j = 0, jn = mbn * ctrlay->n; j < jn; ++j) {
    ctrbuf[j] = sigmoid( mean + randgauss() * dev );
  }
}

}

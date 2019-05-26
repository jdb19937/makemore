#define __MAKEMORE_ENCGEN_CC__ 1

#include <netinet/in.h>

#include <string>
#include <algorithm>

#include "cudamem.hh"
#include "tron.hh"
#include "multitron.hh"
#include "twiddle.hh"
#include "closest.hh"
#include "shibboleth.hh"
#include "shibbomore.hh"
#include "convo.hh"
#include "encgen.hh"
#include "parson.hh"
#include "strutils.hh"
#include "cholo.hh"
#include "normatron.hh"

namespace makemore {

using namespace std;

Encgen::Encgen(const std::string &_dir, unsigned int _mbn) : Project(_dir, _mbn) {
  decay = 0.5; // ???

  assert(mbn > 0);

  assert(config["type"] == "encgen");
  if (config["focus"] == "")
    config["focus"] = "0";
  assert(config["focus"] == "0" || config["focus"] == "1");
  focus = (config["focus"] == "1");

  if (config["ctract"] == "")
    config["ctract"] = "0";
  assert(config["ctract"] == "0" || config["ctract"] == "1");
  ctract = (config["ctract"] == "1");

  char ctxlayfn[4096];
  sprintf(ctxlayfn, "%s/context.lay", dir.c_str());
  ctxlay = new Layout;
  ctxlay->load_file(ctxlayfn);

  char ctrlayfn[4096];
  sprintf(ctrlayfn, "%s/control.lay", dir.c_str());
  ctrlay = new Layout;
  ctrlay->load_file(ctrlayfn);

  char tgtlayfn[4096];
  sprintf(tgtlayfn, "%s/target.lay", dir.c_str());
  tgtlay = new Layout;
  tgtlay->load_file(tgtlayfn);

  char encmapfn[4096], enctopfn[4096];
  sprintf(enctopfn, "%s/enc.top", dir.c_str());
  sprintf(encmapfn, "%s/enc.map", dir.c_str());
  enctop = new Topology;
  enctop->load_file(enctopfn);
  encmap = new Mapfile(encmapfn);
  enc = new Multitron(*enctop, encmap, mbn, ctract);

  char genmapfn[4096], gentopfn[4096];
  sprintf(gentopfn, "%s/gen.top", dir.c_str());
  sprintf(genmapfn, "%s/gen.map", dir.c_str());
  gentop = new Topology;
  gentop->load_file(gentopfn);
  genmap = new Mapfile(genmapfn);
  gen = new Multitron(*gentop, genmap, mbn, false);

  encinlay = new Layout(*tgtlay);
  assert(enc->inn == mbn * encinlay->n);
  assert(enc->outn == mbn * ctrlay->n);

  geninlay = new Layout(*ctxlay);
  *geninlay += *ctrlay;
  assert(gen->inn == mbn * geninlay->n);
  assert(gen->outn == mbn * tgtlay->n);

  cumake(&cuenctgt, enc->outn);
  cumake(&cuencin, enc->inn);
  cumake(&cugentgt, gen->outn);
  cumake(&cugenin, gen->inn);
  cumake(&cugenfin, gen->inn);

  ctxbuf = new double[mbn * ctxlay->n]();
  ctrbuf = new double[mbn * ctrlay->n]();
  tgtbuf = new double[mbn * tgtlay->n]();

  if (focus) {
    cumake(&cutgtlayx, tgtlay->n);
    encude(tgtlay->x, tgtlay->n, cutgtlayx);

    cumake(&cutgtlayy, tgtlay->n);
    encude(tgtlay->y, tgtlay->n, cutgtlayy);
  }

  rounds = 0;
}

Encgen::~Encgen() {
  delete encinlay;
  delete geninlay;

  delete ctxlay;
  delete tgtlay;
  delete ctrlay;

  cufree(cugenin);
  cufree(cugenfin);
  cufree(cuencin);
  cufree(cugentgt);
  cufree(cuenctgt);

  if (focus) {
    cufree(cutgtlayx);
    cufree(cutgtlayy);
  }

  delete[] ctxbuf;
  delete[] ctrbuf;
  delete[] tgtbuf;
}


void Encgen::report(const char *prog) {
  fprintf(
    stderr,
    "%s %s rounds=%u\n"
    "%s %s enc_err2=%g enc_errm=%g\n"
    "%s %s gen_err2=%g gen_errm=%g\n"
    "\n",
    prog, dir.c_str(), rounds,
    prog, dir.c_str(), enc->err2, enc->errm,
    prog, dir.c_str(), gen->err2, gen->errm
  );
}

void Encgen::save() {
  encmap->save();
  genmap->save();
}

void Encgen::load() {
  encmap->load();
  genmap->load();
}

void Encgen::scramble(double dev) {
  for (unsigned int j = 0, jn = mbn * ctrlay->n; j < jn; ++j) {
    ctrbuf[j] = randgauss() * dev;
    if (ctract) {
      ctrbuf[j] = sigmoid(ctrbuf[j]);
    }
  }
}

void Encgen::encode() {
  assert(encinlay->n == tgtlay->n);
  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    encude( tgtbuf + mbi * tgtlay->n, tgtlay->n, cuencin + mbi * encinlay->n);
  }

  const double *cuencout = enc->feed(cuencin, NULL);
  decude(cuencout, enc->outn, ctrbuf);
}

void Encgen::generate() {
  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    encude( ctxbuf + mbi * ctxlay->n, ctxlay->n, cugenin + mbi * geninlay->n + 0);
    encude( ctrbuf + mbi * ctrlay->n, ctrlay->n, cugenin + mbi * geninlay->n + ctxlay->n);
  }

  const double *cugenout = gen->feed(cugenin, NULL);
  decude(cugenout, gen->outn, tgtbuf);
}

void Encgen::burn(double nu, double pi) {
  assert(geninlay->n == ctxlay->n + ctrlay->n);
  assert(encinlay->n == tgtlay->n);
  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    encude(ctxbuf + mbi * ctxlay->n, ctxlay->n, cugenin + mbi * geninlay->n);
    encude(tgtbuf + mbi * tgtlay->n, tgtlay->n, cuencin + mbi * encinlay->n);
  }

  const double *cuencout = enc->feed(cuencin, NULL);
  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    cucopy(cuencout + mbi * ctrlay->n, ctrlay->n, cugenin + mbi * geninlay->n + ctxlay->n);
  }

  cuzero(cugenfin, mbn * geninlay->n);
  gen->feed(cugenin, cugenfin);

  assert(gen->outn == mbn * tgtlay->n);
  encude(tgtbuf, gen->outn, cugentgt);
  gen->target(cugentgt, false);

  if (focus) {
    double *cugenfout = gen->foutput();
    for (unsigned int mbi = 0; mbi < mbn; ++mbi)
      cufocus(cugenfout + mbi * tgtlay->n, cutgtlayx, cutgtlayy, tgtlay->n);
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
}

}

#define __MAKEMORE_ENCODER_CC__ 1

#include <string>
#include <algorithm>

#include "cudamem.hh"
#include "tron.hh"
#include "multitron.hh"
#include "encoder.hh"
#include "parson.hh"
#include "strutils.hh"
#include "imgutils.hh"

namespace makemore {

using namespace std;

Encoder::Encoder(const std::string &_dir, unsigned int _mbn) : Project(_dir, _mbn) {
  assert(mbn > 0);

  assert(config["type"] == "encoder");

  if (config["ctract"] == "")
    config["ctract"] = "0";
  assert(config["ctract"] == "0" || config["ctract"] == "1");
  ctract = (config["ctract"] == "1");

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

  encinlay = new Layout(*tgtlay);
  assert(enc->inn == mbn * encinlay->n);
  assert(enc->outn == mbn * ctrlay->n);

  cumake(&cuenctgt, enc->outn);
  cumake(&cuencin, enc->inn);

  ctrbuf = new double[mbn * ctrlay->n]();
  tgtbuf = new double[mbn * tgtlay->n]();

  rounds = 0;
}

Encoder::~Encoder() {
  delete encinlay;

  delete tgtlay;
  delete ctrlay;

  cufree(cuencin);
  cufree(cuenctgt);

  delete[] ctrbuf;
  delete[] tgtbuf;
}


void Encoder::report(const char *prog) {
  fprintf(
    stderr,
    "%s %s rounds=%u enc_err2=%g enc_errm=%g\n",
    prog, dir.c_str(), rounds, enc->err2, enc->errm
  );
}

void Encoder::save() {
  encmap->save();
}

void Encoder::load() {
  encmap->load();
}

void Encoder::encode(const Partrait &prt, class Parson *prs, class Styler *sty) {
  assert(mbn == 1);
  assert(encinlay->n == tgtlay->n);
  assert(tgtlay->n == prt.w * prt.h * 3);

  rgblab(prt.rgb, tgtlay->n, tgtbuf);
  encude( tgtbuf, tgtlay->n, cuencin);

  const double *cuencout = enc->feed(cuencin, NULL);
  assert(enc->outn == ctrlay->n);
  assert(Parson::ncontrols == ctrlay->n);
  decude(cuencout, enc->outn, prs->controls);

  prt.make_sketch(prs->sketch);

  prs->angle = prt.get_tag("angle", 0.0);
  prs->stretch = prt.get_tag("stretch", 1.0);
  prs->skew = prt.get_tag("skew", 0.0);

  for (auto tag : prt.tags) {
    if (strchr(tag.c_str(), ':'))
      continue;
    prs->add_tag(tag.c_str());
  }

  if (sty) {
    sty->encode(prs->controls, prs);
  }
}

void Encoder::burn(const Partrait &prt, class Generator *gen, double nu, double pi) {
#if 0
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
#endif
}

}

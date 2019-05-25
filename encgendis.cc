#define __MAKEMORE_ENCGENDIS_CC__ 1

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
#include "encgendis.hh"
#include "parson.hh"
#include "strutils.hh"
#include "cholo.hh"
#include "normatron.hh"

namespace makemore {

using namespace std;

static string read_word(FILE *fp, char sep) {
  int c = getc(fp);
  if (c == EOF)
    return "";

  char buf[2];
  buf[0] = (char)c;
  buf[1] = 0;
  string word(buf);

  while (1) {
    c = getc(fp);
    if (c == EOF)
      return "";
    if (c == sep)
      break;
    buf[0] = (char)c;
    word += buf;
  }

  return word;
}

Encgendis::Encgendis(const std::string &_dir, unsigned int _mbn) : Project(_dir, _mbn) {
 decay = 0.5;

  assert(mbn > 0);

  assert(config["type"] == "encgendis");

  char ctxlayfn[4096];
  sprintf(ctxlayfn, "%s/context.lay", dir.c_str());
  ctxlay = new Layout;
  ctxlay->load_file(ctxlayfn);

  char knblayfn[4096];
  sprintf(knblayfn, "%s/knob.lay", dir.c_str());
  knblay = new Layout;
  knblay->load_file(knblayfn);

  char ctrlayfn[4096];
  sprintf(ctrlayfn, "%s/control.lay", dir.c_str());
  ctrlay = new Layout;
  ctrlay->load_file(ctrlayfn);

  char tgtlayfn[4096];
  sprintf(tgtlayfn, "%s/target.lay", dir.c_str());
  tgtlay = new Layout;
  tgtlay->load_file(tgtlayfn);

#if 0
  char distopfn[4096];
  sprintf(distopfn, "%s/dis.top", dir.c_str());
  distop = new Topology;
  distop->load_file(distopfn);

  char dismapfn[4096];
  sprintf(dismapfn, "%s/dis.map", dir.c_str());
  dismap = new Mapfile(dismapfn);
  dis = new Multitron(*distop, dismap, mbn * 2, false);

  assert(dis->outn == mbn * 2);
  assert(dis->inn == mbn * 2 * (ctxlay->n + tgtlay->n));
#endif

  char encmapfn[4096], enctopfn[4096];
  sprintf(enctopfn, "%s/enc.top", dir.c_str());
  sprintf(encmapfn, "%s/enc.map", dir.c_str());
  enctop = new Topology;
  enctop->load_file(enctopfn);
  encmap = new Mapfile(encmapfn);
  enc = new Multitron(*enctop, encmap, mbn, false);

#if 0
  char inencmapfn[4096], inenctopfn[4096];
  sprintf(inenctopfn, "%s/inenc.top", dir.c_str());
  sprintf(inencmapfn, "%s/inenc.map", dir.c_str());
  inenctop = new Topology;
  inenctop->load_file(inenctopfn);
  inencmap = new Mapfile(inencmapfn);
  inenc = new Multitron(*inenctop, inencmap, mbn, true);
  assert(inenc->inn == mbn * ctrlay->n);
  assert(inenc->outn == mbn * knblay->n);
#endif

  char genmapfn[4096], gentopfn[4096];
  sprintf(gentopfn, "%s/gen.top", dir.c_str());
  sprintf(genmapfn, "%s/gen.map", dir.c_str());
  gentop = new Topology;
  gentop->load_file(gentopfn);
  genmap = new Mapfile(genmapfn);
  gen = new Multitron(*gentop, genmap, mbn, false);

#if 0
  char ingenmapfn[4096], ingentopfn[4096];
  sprintf(ingentopfn, "%s/ingen.top", dir.c_str());
  sprintf(ingenmapfn, "%s/ingen.map", dir.c_str());
  ingentop = new Topology;
  ingentop->load_file(ingentopfn);
  ingenmap = new Mapfile(ingenmapfn);
  ingen = new Multitron(*ingentop, ingenmap, mbn, true);
  assert(ingen->inn == mbn * knblay->n);
  assert(ingen->outn == mbn * ctrlay->n);
#endif

  encinlay = new Layout(*tgtlay);
  assert(enc->inn == mbn * encinlay->n);
  assert(enc->outn == mbn * ctrlay->n);

  geninlay = new Layout(*ctxlay);
  *geninlay += *ctrlay;
  assert(gen->inn == mbn * geninlay->n);
  assert(gen->outn == mbn * tgtlay->n);

  disinlay = new Layout(*ctxlay);
#if 0
  *disinlay += *ctrlay;
#endif
  *disinlay += *tgtlay;

  cumake(&cuenctgt, enc->outn);
  cumake(&cuencin, enc->inn);
  cumake(&cugentgt, gen->outn);
  cumake(&cugenin, gen->inn);
  cumake(&cugenfin, gen->inn);

  ctxbuf = new double[mbn * ctxlay->n]();
  ctrbuf = new double[mbn * ctrlay->n]();
  knbbuf = new double[mbn * knblay->n]();
  tgtbuf = new double[mbn * tgtlay->n]();

  cumake(&cutgtlayx, tgtlay->n);
  encude(tgtlay->x, tgtlay->n, cutgtlayx);

  cumake(&cutgtlayy, tgtlay->n);
  encude(tgtlay->y, tgtlay->n, cutgtlayy);

  rounds = 0;
}

Encgendis::~Encgendis() {
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

  delete[] ctxbuf;
  delete[] ctrbuf;
  delete[] tgtbuf;
}


void Encgendis::report(const char *prog) {
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

void Encgendis::save() {
  encmap->save();
  genmap->save();
//  dismap->save();
}

void Encgendis::load() {
  encmap->load();
  genmap->load();
  dismap->load();
}

void Encgendis::observe(double mu, double yo, double wu, const double *realness) {
++rounds;
  
  {
    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      encude( ctxbuf + mbi * ctxlay->n, ctxlay->n, cugenin + mbi * geninlay->n + 0);
      encude( ctrbuf + mbi * ctrlay->n, ctrlay->n, cugenin + mbi * geninlay->n + ctxlay->n);
    }
    const double *cugenout = gen->feed(cugenin, NULL);
    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      encude(  ctxbuf + mbi * ctxlay->n, ctxlay->n, cudisin + mbi * disinlay->n + 0);

#if 0
      encude( ctrbuf + mbi * ctxlay->n, ctrlay->n, cudisin + mbi * disinlay->n + ctxlay->n);
      cucopy(cugenout + mbi * tgtlay->n, tgtlay->n, cudisin + mbi * disinlay->n + ctxlay->n + ctrlay->n);
#else
      cucopy(cugenout + mbi * tgtlay->n, tgtlay->n, cudisin + mbi * disinlay->n + ctxlay->n);
#endif

    }
  
  
#if 0
    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      encude( ctxbuf + mbi * ctxlay->n, ctxlay->n, cuencin + mbi * encinlay->n + 0);
      encude( tgtbuf + mbi * tgtlay->n, tgtlay->n, cuencin + mbi * encinlay->n + ctxlay->n);
    }
    const double *cuencout = enc->feed(cuencin, NULL);
#endif

    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      encude(  ctxbuf + mbi * ctxlay->n, ctxlay->n, cudisin + (mbi + mbn) * disinlay->n + 0);
#if 0
      cucopy(cuencout + mbi * ctrlay->n, ctrlay->n, cudisin + (mbi + mbn) * disinlay->n + ctxlay->n);
      encude(  tgtbuf + mbi * tgtlay->n, tgtlay->n, cudisin + (mbi + mbn) * disinlay->n + ctxlay->n + ctrlay->n);
#else
      encude(  tgtbuf + mbi * tgtlay->n, tgtlay->n, cudisin + (mbi + mbn) * disinlay->n + ctxlay->n);
#endif
    }
  
  
    cuzero(cudisfin, dis->inn);
    const double *cudisout = dis->feed(cudisin, cudisfin);

    double disbuf[mbn * 2];
    decude(cudisout, mbn * 2, disbuf);

    for (unsigned int mbi = 0, dmbn = 2 * mbn; mbi < dmbn; ++mbi) {
      distgt[mbi] = mbi < mbn ? 1.0 : 0.0;
    }

    encude(distgt, dis->outn, cudistgt);

    dis->target(cudistgt, false);
    dis->train(0);
  
    double *cugenfout = gen->foutput();
    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
#if 0
      cucopy(cudisfin + mbi * disinlay->n + ctxlay->n + ctrlay->n, tgtlay->n, cugenfout + mbi * tgtlay->n);
#else
      cucopy(cudisfin + mbi * disinlay->n + ctxlay->n, tgtlay->n, cugenfout + mbi * tgtlay->n);
#endif
    }
  
#if 0
    double *cuencfout = enc->foutput();
    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      cucopy(cudisfin + (mbi + mbn) * disinlay->n + ctxlay->n, ctrlay->n, cuencfout + mbi * ctrlay->n);
    }
  
    enc->update_stats();
    enc->train(yo);
#endif
  
    gen->update_stats();
//    if (dis->err2 > 0.01)
//      wu *= 1e-10;

double old_b1 = adam_b1;
double old_b2 = adam_b2;
double old_b3 = adam_b3;
double old_eps = adam_eps;

//adam_b2 = 0.99;
//adam_eps = 1.0;


//if (rounds % 4 == 0)
    gen->train(wu);

adam_b1 = old_b1;
adam_b2 = old_b2;
adam_b3 = old_b3;
adam_eps = old_eps;




    cudisout = dis->feed(cudisin, NULL);
    decude(cudisout, mbn * 2, disbuf);

    for (unsigned int mbi = 0, dmbn = 2 * mbn; mbi < dmbn; ++mbi) {
      distgt[mbi] = (mbi < mbn) ? disbuf[mbi] : realness[mbi - mbn];
    }
    encude(distgt, dis->outn, cudistgt);
  
    dis->target(cudistgt);
    dis->train(mu);
  }
}

void Encgendis::scramble(double dev) {
  for (unsigned int j = 0, jn = mbn * ctrlay->n; j < jn; ++j) {
    ctrbuf[j] = randgauss() * dev;
  }
}

void Encgendis::encode() {
  assert(encinlay->n == tgtlay->n);
  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    encude( tgtbuf + mbi * tgtlay->n, tgtlay->n, cuencin + mbi * encinlay->n);
  }

  const double *cuencout = enc->feed(cuencin, NULL);
  decude(cuencout, enc->outn, ctrbuf);
}

void Encgendis::inencode() {
  encude(ctrbuf, mbn * ctrlay->n, cuinencin);
  const double *cuinencout = inenc->feed(cuinencin, NULL);
  decude(cuinencout, inenc->outn, knbbuf);
}



void Encgendis::segment() {
}

void Encgendis::generate() {
  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    encude( ctxbuf + mbi * ctxlay->n, ctxlay->n, cugenin + mbi * geninlay->n + 0);
    encude( ctrbuf + mbi * ctrlay->n, ctrlay->n, cugenin + mbi * geninlay->n + ctxlay->n);
  }

  const double *cugenout = gen->feed(cugenin, NULL);
  decude(cugenout, gen->outn, tgtbuf);
}

void Encgendis::ingenerate() {
  encude(knbbuf, mbn * knblay->n, cuingenin);
  const double *cuingenout = ingen->feed(cuingenin, NULL);
  decude(cuingenout, ingen->outn, ctrbuf);
}


void Encgendis::inscramble(double *knobs, unsigned int n, Cholo *cholo) {
#if 1
  double *tmp = new double[knblay->n * mbn];
#endif
  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {

#if 1
    for (unsigned int j = mbi * knblay->n, jn = j + knblay->n; j < jn; ++j) {
      unsigned int k = randuint() % n;
      tmp[j] = knobs[k * knblay->n + j] * 1.0;
    }

    cholo->generate(tmp + mbi * knblay->n, knbbuf + mbi * knblay->n);

    for (unsigned int j = mbi * knblay->n, jn = j + knblay->n; j < jn; ++j) {
      knbbuf[j] = sigmoid(knbbuf[j]);
    }
#endif

#if 0
    for (unsigned int j = mbi * ctrlay->n, jn = j + knblay->n; j < jn; ++j) {
      knbbuf[j] = sigmoid(randgauss());
    }
#endif

  }
#if 1
  delete[] tmp;
#endif
}

static unsigned int *used = new unsigned int[10000]();

void Encgendis::inconc(const double *cuconc, unsigned int n, double nu, double pi) {
  ++rounds;

if (mbn == 1) {
  unsigned int k = randuint() % n;
  ingen->feed(cuconc + k * (knblay->n + ctrlay->n), NULL);
  ingen->target(cuconc + k * (knblay->n + ctrlay->n) + knblay->n);
  ingen->train(pi);
  return;
}

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    unsigned int k = randuint() % n;
    cucopy(cuconc + k * (knblay->n + ctrlay->n), knblay->n, cuingenin + mbi * knblay->n);
    cucopy(cuconc + k * (knblay->n + ctrlay->n) + knblay->n, ctrlay->n, cuinencin + mbi * ctrlay->n);
  }

  ingen->feed(cuingenin, NULL);

  ingen->target(cuinencin);
  ingen->train(pi);
}

void Encgendis::inburn(const double *cusamp, unsigned int n, double nu, double pi) {

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    unsigned int i = randuint() % n;
    cucopy(cusamp + i * ctrlay->n, ctrlay->n, cuinencin + mbi * ctrlay->n);
  }
  assert(inenc->inn == mbn * ctrlay->n);
  assert(ingen->outn == mbn * ctrlay->n);
  assert(ingen->inn == inenc->outn);
  assert(ingen->outn == inenc->inn);

  const double *cuinencout = inenc->feed(cuinencin, NULL);
  ingen->feed(cuinencout, inenc->foutput());

  ingen->target(cuinencin);
  ingen->train(pi);

  inenc->update_stats();
  inenc->train(nu);
}

void Encgendis::burn(double nu, double pi) {

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

#if 1
  double *cugenfout = gen->foutput();
  for (unsigned int mbi = 0; mbi < mbn; ++mbi)
    cufocus(cugenfout + mbi * tgtlay->n, cutgtlayx, cutgtlayy, tgtlay->n);
#endif

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


void Encgendis::burngen(double pi) {
  assert(geninlay->n == ctxlay->n + ctrlay->n);
  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    encude(ctxbuf + mbi * ctxlay->n, ctxlay->n, cugenin + mbi * geninlay->n + 0);
    encude(ctrbuf + mbi * ctrlay->n, tgtlay->n, cugenin + mbi * geninlay->n + ctxlay->n);
  }
  assert(gen->outn == mbn * tgtlay->n);
  encude(tgtbuf, gen->outn, cugentgt);

  gen->feed(cugenin, NULL);
  gen->target(cugentgt);
  gen->train(pi);
}

void Encgendis::burnenc(double nu) {
  assert(encinlay->n == ctxlay->n + tgtlay->n);
  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    encude(ctxbuf + mbi * ctxlay->n, ctxlay->n, cuencin + mbi * encinlay->n + 0);
    encude(tgtbuf + mbi * tgtlay->n, tgtlay->n, cuencin + mbi * encinlay->n + ctxlay->n);
  }
  assert(enc->outn == mbn * ctrlay->n);
  encude(ctrbuf, enc->outn, cuenctgt);

  enc->feed(cuencin, NULL);
  enc->target(cuenctgt);
  enc->train(nu);
}

}

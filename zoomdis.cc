#define __MAKEMORE_ZOOMDIS_CC__ 1

#include <string>
#include <algorithm>

#include "cudamem.hh"
#include "tron.hh"
#include "multitron.hh"
#include "zoomdis.hh"
#include "parson.hh"
#include "strutils.hh"
#include "imgutils.hh"
#include "zoomgen.hh"
#include "pic.hh"
#include "partrait.hh"
#include "numutils.hh"

namespace makemore {

using namespace std;

Zoomdis::Zoomdis(const std::string &_dir, unsigned int _mbn) : Project(_dir, _mbn) {
  assert(mbn > 0);

  assert(config["type"] == "zoomdis");

  char dismapfn[4096];
  sprintf(dismapfn, "%s/dis.map", dir.c_str());
  dismap = new Mapfile(dismapfn);
  dis = new Supertron(dismap);

  cumake(&cudisin, dis->inn);
  cumake(&cudistgt, dis->outn);

  inpbuf = new double[mbn * dis->inn]();

  rounds = 0;
}

Zoomdis::~Zoomdis() {
  cufree(cudisin);
  cufree(cudistgt);

  delete[] inpbuf;
}


void Zoomdis::report(const char *prog) {
  fprintf(
    stderr,
    "%s %s rounds=%u dis_err2=%g dis_errm=%g\n",
    prog, dir.c_str(), rounds,
    dis->err2, dis->errm
  );
}

void Zoomdis::save() {
  dismap->save();
}

void Zoomdis::load() {
  dismap->load();
}

static void addnoise(double *cudat, unsigned int n, double dev) {
  if (dev == 0.0)
    return;

  double *noise = new double[n];
  for (unsigned int j = 0; j < n; ++j)
    noise[j] = randgauss() * dev;
  double *cunoise;
  cumake(&cunoise, n);
  encude(noise, n, cunoise);
  delete[] noise;

  cuaddvec(cudat, cunoise, n, cudat);
  cufree(cunoise);
}

double Zoomdis::score(const class Partrait &prt, double noise) {
  assert(mbn == 1);
  assert(dis->inn == prt.w * prt.h * 3);

  btodv(prt.rgb, inpbuf, dis->inn);
  encude( inpbuf, dis->inn, cudisin);

  addnoise(cudisin, dis->inn, noise);

  const double *cudisout = dis->feed(cudisin);
  double sc;

  decude(cudisout, 1, &sc);
  return sigmoid(sc);
  return sc;
}

double Zoomdis::score(const class Zoomgen *gen, double noise) {
  assert(mbn == 1);
  assert(gen->tgtlay->n == dis->inn);

  addnoise(gen->gen->output(), dis->inn, noise);

  const double *cudisout = dis->feed(gen->gen->output(), gen->gen->foutput());

  double sc;
  decude(cudisout, 1, &sc);
  return sigmoid(sc);
  return sc;
}

void Zoomdis::burnreal(double nu) {
  assert(dis->outn == 1);

  double p;
  decude(dis->output(), 1, &p);

  // double loss = 1.0 / (p > MINP ? p : MINP);
  double loss = 1.0 - sigmoid(p);

  double sc1 = p + loss;

  encude(&sc1, 1, cudistgt);

  dis->target(cudistgt);
  dis->update_stats();
  dis->train(nu);
}

void Zoomdis::burnfake(double nu) {
  assert(dis->outn == 1);

  double p;
  decude(dis->output(), 1, &p);

  // double loss = -1.0 / (1.0 - (p < (1.0 - MINP) ? p : (1.0 - MINP)));
  double loss = 0.0 - sigmoid(p);

  double sc1 = p + loss;

  encude(&sc1, 1, cudistgt);

  dis->target(cudistgt);
  dis->update_stats();
  dis->train(nu);
}

void Zoomdis::testfake() {
  assert(dis->outn == 1);

  double p;
  decude(dis->output(), 1, &p);

  double loss = 1.0 - sigmoid(p);

  double sc1 = p + loss;

  encude(&sc1, 1, cudistgt);

  dis->target(cudistgt);
  dis->train(0);
}



void Zoomdis::observe(const Partrait *prt0, Zoomgen *gen, const Partrait *prt1, double nu) {
//  if (dis->err2 < 0.3)
//    nu = 1e-24;

  bool lswap = (randuint() % 64 == 0);
  double noise = 0.01;

  gen->generate(*prt0);

  double sc0 = score(gen, noise);
  if (lswap)
    burnreal(nu);
  else
    burnfake(nu);

  double sc1 = score(*prt1, noise);
  if (lswap)
    burnfake(nu);
  else
    burnreal(nu);

fprintf(stderr, "sc0=%lf sc1=%lf err2=%lf\n", sc0, sc1, dis->err2);
}



}

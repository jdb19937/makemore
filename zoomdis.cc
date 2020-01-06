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

namespace makemore {

using namespace std;

Zoomdis::Zoomdis(const std::string &_dir, unsigned int _mbn) : Project(_dir, _mbn) {
  assert(mbn > 0);

  assert(config["type"] == "zoomdis");

  char inplayfn[4096];
  sprintf(inplayfn, "%s/input.lay", dir.c_str());
  inplay = new Layout;
  inplay->load_file(inplayfn);

  char dismapfn[4096];
  sprintf(dismapfn, "%s/dis.map", dir.c_str());
  dismap = new Mapfile(dismapfn);
  dis = new Supertron(dismap);

  assert(dis->inn == inplay->n);
  assert(dis->outn == 1);

  cumake(&cudisin, dis->inn);
  cumake(&cudistgt, 1);

  inpbuf = new double[mbn * inplay->n]();

  rounds = 0;
}

Zoomdis::~Zoomdis() {
  delete inplay;

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

double Zoomdis::score(const class Partrait &prt) {
  assert(mbn == 1);
  assert(inplay->n == prt.w * prt.h * 3);

  rgblab(prt.rgb, inplay->n, inpbuf);
  encude( inpbuf, inplay->n, cudisin);
  const double *cudisout = dis->feed(cudisin);
  double sc;

  decude(cudisout, 1, &sc);
  return sc;
}

double Zoomdis::score(const class Zoomgen *gen) {
  assert(mbn == 1);
  assert(gen->tgtlay->n == inplay->n);
  const double *cudisout = dis->feed(gen->gen->output(), gen->gen->foutput());

  double sc;
  decude(cudisout, 1, &sc);
  return sc;
}

void Zoomdis::burn(double sc, double nu) {
  encude(&sc, 1, cudistgt);
  dis->target(cudistgt);

  if (nu > 0)
    dis->update_stats();
  dis->train(nu);
}

void Zoomdis::observe(const Partrait *prt0, Zoomgen *gen, const Partrait *prt1, double nu) {
  gen->generate(*prt0);

  double sc0 = score(gen);
  burn(0.0, nu);

  double sc1 = score(*prt1);
  burn(1.0, nu);
}



}

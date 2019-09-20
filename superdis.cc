#define __MAKEMORE_SUPERDIS_CC__ 1

#include <string>
#include <algorithm>

#include "cudamem.hh"
#include "tron.hh"
#include "multitron.hh"
#include "superdis.hh"
#include "parson.hh"
#include "strutils.hh"
#include "imgutils.hh"
#include "supergen.hh"
#include "superenc.hh"

namespace makemore {

using namespace std;

Superdis::Superdis(const std::string &_dir, unsigned int _mbn) : Project(_dir, _mbn) {
  assert(mbn > 0);

  assert(config["type"] == "superdis");

  char inplayfn[4096];
  sprintf(inplayfn, "%s/input.lay", dir.c_str());
  inplay = new Layout;
  inplay->load_file(inplayfn);

#if 0
  char clsmapfn[4096];
  sprintf(clsmapfn, "%s/cls.map", dir.c_str());
  clsmap = new Mapfile(clsmapfn);
  cls = new Supertron(clsmap);
#endif

  char dismapfn[4096];
  sprintf(dismapfn, "%s/dis.map", dir.c_str());
  dismap = new Mapfile(dismapfn);
  dis = new Supertron(dismap);

#if 0
  assert(cls->inn == inplay->n);
  assert(cls->outn == dis->inn);
#else
  assert(dis->inn == inplay->n);
#endif
  assert(dis->outn == 1);

#if 0
  cumake(&cuclsin, cls->inn);
  cumake(&cuclstgt, cls->outn);
#endif
  cumake(&cudisin, dis->inn);
  cumake(&cudistgt, 1);

  inpbuf = new double[mbn * inplay->n]();

  rounds = 0;
}

Superdis::~Superdis() {
  delete inplay;

  cufree(cudisin);
#if 0
  cufree(cuclsin);
  cufree(cuclstgt);
#endif
  cufree(cudistgt);

  delete[] inpbuf;
}


void Superdis::report(const char *prog) {
  fprintf(
    stderr,
#if 0
    "%s %s rounds=%u cls_err2=%g cls_errm=%g\n"
#endif
    "%s %s rounds=%u dis_err2=%g dis_errm=%g\n",
#if 0
    prog, dir.c_str(), rounds,
    cls->err2, cls->errm,
#endif
    prog, dir.c_str(), rounds,
    dis->err2, dis->errm
  );
}

void Superdis::save() {
  //clsmap->save();
  dismap->save();
}

void Superdis::load() {
  //clsmap->load();
  dismap->load();
}

double Superdis::score(const class Partrait &prt) {
  assert(mbn == 1);
  assert(inplay->n == prt.w * prt.h * 3);

  rgblab(prt.rgb, inplay->n, inpbuf);

  double d = 0.5 / 256.0;
  for (unsigned int j = 0; j < inplay->n; ++j)
    inpbuf[j] += randrange(-d, d);

#if 0
  encude( inpbuf, inplay->n, cuclsin);
  const double *cuclsout = cls->feed(cuclsin, NULL);
  const double *cudisout = dis->feed(cuclsout, cls->foutput());
#endif
  encude( inpbuf, inplay->n, cudisin);
  const double *cudisout = dis->feed(cudisin);
  double sc;

  decude(cudisout, 1, &sc);
  return sc;
}

#if 0
void Superdis::classify(const class Partrait &prt, double *clsbuf) {
  assert(mbn == 1);
  assert(inplay->n == prt.w * prt.h * 3);

  rgblab(prt.rgb, inplay->n, inpbuf);

  double d = 0.5 / 256.0;
  for (unsigned int j = 0; j < inplay->n; ++j)
    inpbuf[j] += randrange(-d, d);

  encude( inpbuf, inplay->n, cuclsin);
  const double *cuclsout = cls->feed(cuclsin, NULL);

  decude(cuclsout, cls->outn, clsbuf);
}
#endif

double Superdis::score(const class Supergen *gen) {
  assert(mbn == 1);
  assert(gen->tgtlay->n == inplay->n);

#if 0
  const double *cuclsout = cls->feed(gen->gen->output(), gen->gen->foutput());
  const double *cudisout = dis->feed(cls->output(), cls->foutput());
#endif
  const double *cudisout = dis->feed(gen->gen->output(), gen->gen->foutput());

  double sc;
  decude(cudisout, 1, &sc);
  return sc;
}

void Superdis::burn(double sc, double nu) {

//  double osc;
//  decude(dis->output(), 1, &osc);
//  double dsc = (sc - sigmoid(osc));
//  encude(&dsc, 1, dis->foutput());

  encude(&sc, 1, cudistgt);
  dis->target(cudistgt);

  if (nu > 0)
    dis->update_stats();
  dis->train(nu);

//  if (nu > 0)
//    cls->update_stats();
//  cls->train(nu);
}

void Superdis::observe(const Partrait *prt0, Superenc *enc, Supergen *gen, const Partrait *prt1, double nu) {

  double *ctr;
  cumake(&ctr, enc->ctrlay->n);

  enc->encode(*prt0, ctr);
  gen->generate(ctr, NULL);

  cufree(ctr);

  double sc0 = score(gen);
  burn(0.0, nu);

  double sc1 = score(*prt1);
  burn(1.0, nu);

//fprintf(stderr, "(%lf,%lf)\n", sc0, sc1);
}


}

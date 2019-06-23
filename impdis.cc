#define __MAKEMORE_IMPDIS_CC__ 1

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
#include "impdis.hh"
#include "parson.hh"
#include "strutils.hh"
#include "cholo.hh"
#include "normatron.hh"
#include "impdis.hh"

namespace makemore {

using namespace std;

Impdis::Impdis(const std::string &_dir, unsigned int _mbn) : Project(_dir, _mbn) {
  assert(mbn > 0);

  assert(config["type"] == "impdis");
  if (config["focus"] == "")
    config["focus"] = "0";
  assert(config["focus"] == "0" || config["focus"] == "1");
  focus = (config["focus"] == "1");

  char tgtlayfn[4096];
  sprintf(tgtlayfn, "%s/target.lay", dir.c_str());
  tgtlay = new Layout;
  tgtlay->load_file(tgtlayfn);

  char impmapfn[4096], imptopfn[4096];
  sprintf(imptopfn, "%s/imp.top", dir.c_str());
  sprintf(impmapfn, "%s/imp.map", dir.c_str());
  imptop = new Topology;
  imptop->load_file(imptopfn);
  impmap = new Mapfile(impmapfn);
  imp = new Multitron(*imptop, impmap, mbn, false);

  char dismapfn[4096], distopfn[4096];
  sprintf(distopfn, "%s/dis.top", dir.c_str());
  sprintf(dismapfn, "%s/dis.map", dir.c_str());
  distop = new Topology;
  distop->load_file(distopfn);
  dismap = new Mapfile(dismapfn);
  dis = new Multitron(*distop, dismap, mbn, false);

  assert(dis->outn == mbn);

  cumake(&cuimptgt, imp->outn);
  cumake(&cuimpin, imp->inn);
  cumake(&cuimpfin, imp->inn);
  cumake(&cudistgt, dis->outn);
  cumake(&cudisin, dis->inn);

  tgtbuf = new double[mbn * tgtlay->n]();
  inbuf = new double[mbn * tgtlay->n]();

  if (focus) {
    cumake(&cutgtlayx, tgtlay->n);
    encude(tgtlay->x, tgtlay->n, cutgtlayx);

    cumake(&cutgtlayy, tgtlay->n);
    encude(tgtlay->y, tgtlay->n, cutgtlayy);
  }

  rounds = 0;
}

Impdis::~Impdis() {
  delete tgtlay;

  cufree(cuimpin);
  cufree(cuimptgt);
  cufree(cuimpfin);
  cufree(cudisin);
  cufree(cudistgt);

  if (focus) {
    cufree(cutgtlayx);
    cufree(cutgtlayy);
  }

  delete[] tgtbuf;
}


void Impdis::report(const char *prog) {
  fprintf(
    stderr,
    "%s %s rounds=%u\n"
    "%s %s imp_err2=%g imp_errm=%g\n"
    "%s %s dis_err2=%g dis_errm=%g\n"
    "\n",
    prog, dir.c_str(), rounds,
    prog, dir.c_str(), imp->err2, imp->errm,
    prog, dir.c_str(), dis->err2, dis->errm
  );
}

void Impdis::save() {
  impmap->save();
  dismap->save();
}

void Impdis::load() {
  impmap->load();
  dismap->load();
}

void Impdis::burn(double pi) {
  assert(imp->inn == tgtlay->n * mbn);
  assert(imp->outn == tgtlay->n * mbn);

  encude(inbuf, imp->inn, cuimpin);
  imp->feed(cuimpin, NULL);

  encude(tgtbuf, imp->outn, cuimptgt);
  cusubvec(cuimptgt, cuimpin, mbn * tgtlay->n, cuimptgt);
  imp->target(cuimptgt, false);

  if (focus) {
    double *cuimpfout = imp->foutput();
    for (unsigned int mbi = 0; mbi < mbn; ++mbi)
      cufocus(cuimpfout + mbi * tgtlay->n, cutgtlayx, cutgtlayy, tgtlay->n);
  }

  imp->update_stats();
  imp->train(pi);
}

void Impdis::observe(double mu, double xi) {
  encude(tgtbuf, mbn * tgtlay->n, cudisin);

  double *blend = new double[mbn];
  for (unsigned int mbi = 0; mbi < mbn; ++mbi)
    blend[mbi] = randrange(0.0, 1.0);
  encude(blend, mbn, cudistgt);

  encude(inbuf, imp->inn, cuimpin);
  imp->feed(cuimpin, NULL);

  double *tmp;
  cumake(&tmp, imp->outn);
  cucopy(imp->output(), imp->outn, tmp);
  cuaddvec(tmp, cuimpin, mbn * tgtlay->n, tmp);

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    cumuld(tmp + mbi * tgtlay->n, 1.0 - blend[mbi], tgtlay->n, tmp + mbi * tgtlay->n);
    cumuld(cudisin + mbi * tgtlay->n, blend[mbi], tgtlay->n, cudisin + mbi * tgtlay->n);
    cuaddvec(cudisin + mbi * tgtlay->n, tmp + mbi * tgtlay->n, tgtlay->n, cudisin + mbi * tgtlay->n);
  }

  dis->feed(cudisin, NULL);
  dis->target(cudistgt);
  dis->train(xi);

  for (unsigned int mbi = 0; mbi < mbn; ++mbi)
    blend[mbi] = 1.0;
  encude(blend, mbn, cudistgt);

  cucopy(imp->output(), imp->outn, tmp);
  cuaddvec(tmp, cuimpin, mbn * tgtlay->n, tmp);

  dis->feed(tmp, imp->foutput());
  dis->target(cudistgt, false);
  dis->train(0);

  imp->update_stats();
  imp->train(mu);

  delete[] blend;
  cufree(tmp);
}

void Impdis::improve() {
  encude(inbuf, imp->inn, cuimpin);
  const double *cuimpout = imp->feed(cuimpin, NULL);
  cuaddvec(cuimpin, cuimpout, tgtlay->n, cuimpin);
  decude(cuimpin, imp->outn, tgtbuf);
}

}

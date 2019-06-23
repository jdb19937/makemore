#define __MAKEMORE_AUTOMASKER_CC__ 1

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
#include "parson.hh"
#include "strutils.hh"
#include "cholo.hh"
#include "normatron.hh"
#include "automasker.hh"
#include "partrait.hh"
#include "numutils.hh"

namespace makemore {

using namespace std;

Automasker::Automasker(const std::string &_dir) : Project(_dir, 1) {
  assert(mbn == 1);
  assert(config["type"] == "automasker");

  char mskoutlayfn[4096];
  sprintf(mskoutlayfn, "%s/mskoutput.lay", dir.c_str());
  mskoutlay = new Layout;
  mskoutlay->load_file(mskoutlayfn);

  char mskinlayfn[4096];
  sprintf(mskinlayfn, "%s/mskinput.lay", dir.c_str());
  mskinlay = new Layout;
  mskinlay->load_file(mskinlayfn);

  char mskmapfn[4096], msktopfn[4096];
  sprintf(msktopfn, "%s/msk.top", dir.c_str());
  sprintf(mskmapfn, "%s/msk.map", dir.c_str());
  msktop = new Topology;
  msktop->load_file(msktopfn);
  mskmap = new Mapfile(mskmapfn);
  msk = new Multitron(*msktop, mskmap, mbn, false);

  assert(msk->outn == mskoutlay->n);
  assert(mskoutlay->n * 3 == mskinlay->n);

  cumake(&cumsktgt, msk->outn);
  cumake(&cumskin, msk->inn);

  tgtbuf = new double[msk->inn];

  rounds = 0;
}

Automasker::~Automasker() {
  delete msk;
  delete mskmap;
  delete msktop;

  delete mskinlay;
  delete mskoutlay;
 
  delete[] tgtbuf;

  cufree(cumskin);
  cufree(cumsktgt);
}



void Automasker::report(const char *prog) {
  fprintf(
    stderr,
    "%s %s rounds=%u\n"
    "%s %s msk_err2=%g msk_errm=%g\n"
    "\n",
    prog, dir.c_str(), rounds,
    prog, dir.c_str(), msk->err2, msk->errm
  );
}

void Automasker::save() {
  mskmap->save();
}

void Automasker::load() {
  mskmap->load();
}

void Automasker::observe(const Partrait &prt, double mu) {
  assert(mbn == 1);
  assert(msk->inn == prt.w * prt.h * 3);
  assert(msk->outn == prt.w * prt.h);
  assert(prt.rgb);
  assert(prt.alpha);

  btodv(prt.rgb, tgtbuf, msk->inn);
  encude(tgtbuf, msk->inn, cumskin);
  msk->feed(cumskin, NULL);

  btodv(prt.alpha, tgtbuf, msk->outn);
  encude(tgtbuf, msk->outn, cumsktgt);
  msk->target(cumsktgt, false);

  msk->update_stats();
  msk->train(mu);
}

void Automasker::automask(Partrait *prt) {
  assert(mbn == 1);
  assert(msk->inn == prt->w * prt->h * 3);
  assert(msk->inn == msk->outn * 3);

  btodv(prt->rgb, tgtbuf, msk->inn);
  encude(tgtbuf, msk->inn, cumskin);
  const double *cumskout = msk->feed(cumskin, NULL);

  decude(cumskout, msk->outn, tgtbuf);

  if (!prt->alpha)
    prt->alpha = new uint8_t[prt->w * prt->h];
  dtobv(tgtbuf, prt->alpha, msk->outn);
}

}

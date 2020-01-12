#define __MAKEMORE_ZOOMGEN_CC__ 1

#include <string>
#include <algorithm>

#include "cudamem.hh"
#include "tron.hh"
#include "multitron.hh"
#include "encoder.hh"
#include "zoomgen.hh"
#include "parson.hh"
#include "strutils.hh"
#include "imgutils.hh"
#include "cholo.hh"
#include "numutils.hh"
#include "zoomdis.hh"
#include "pic.hh"

namespace makemore {

using namespace std;

Zoomgen::Zoomgen(const std::string &_dir, unsigned int _mbn) : Project(_dir, _mbn) {
  assert(mbn > 0);

  assert(config["type"] == "zoomgen");
  if (config["focus"] == "")
    config["focus"] = "0";
  assert(config["focus"] == "0" || config["focus"] == "1");
  focus = (config["focus"] == "1");

  if (config["ctract"] == "")
    config["ctract"] = "0";
  assert(config["ctract"] == "0" || config["ctract"] == "1");
  ctract = (config["ctract"] == "1");

  is_rgb = false;
  if (config["color"] == "")
    config["color"] = "lab";
  assert(config["color"] == "lab" || config["color"] == "rgb");
  if (config["color"] == "rgb")
    is_rgb = true;

  ctxtype = 0;
  ctxtype = atoi(config["ctxtype"].c_str());

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

  char genmapfn[4096];
  sprintf(genmapfn, "%s/gen.map", dir.c_str());
  genmap = new Mapfile(genmapfn);
  gen = new Supertron(genmap);

  geninlay = new Layout(*ctxlay);
  *geninlay += *ctrlay;
  assert(gen->inn == mbn * geninlay->n);
  assert(gen->outn == mbn * tgtlay->n);

  cumake(&cugentgt, gen->outn);
  cumake(&cugenin, gen->inn);
  cumake(&cugenfin, gen->inn);

  ctxbuf = new double[mbn * ctxlay->n]();
  ctrbuf = new double[mbn * ctrlay->n]();
  tgtbuf = new double[mbn * tgtlay->n]();
  buf = new double[mbn * tgtlay->n]();

  if (focus) {
    cumake(&cutgtlayx, tgtlay->n);
    encude(tgtlay->x, tgtlay->n, cutgtlayx);

    cumake(&cutgtlayy, tgtlay->n);
    encude(tgtlay->y, tgtlay->n, cutgtlayy);
  }

  zone = new Zone(dir + "/train.zone");

  rounds = 0;
}

Zoomgen::~Zoomgen() {
  delete geninlay;

  delete ctxlay;
  delete tgtlay;
  delete ctrlay;

  cufree(cugenin);
  cufree(cugenfin);

  if (focus) {
    cufree(cutgtlayx);
    cufree(cutgtlayy);
  }

  delete[] ctxbuf;
  delete[] ctrbuf;
  delete[] tgtbuf;
  delete[] buf;
}


void Zoomgen::report(const char *prog) {
  fprintf(
    stderr,
    "%s %s rounds=%u gen_err2=%g gen_errm=%g\n",
    prog, dir.c_str(), rounds, gen->err2, gen->errm
  );
}

void Zoomgen::save() {
  genmap->save();
}

void Zoomgen::load() {
  genmap->load();
}

void Zoomgen::generate(const Partrait &pic0, Partrait *outpic) {
  assert(mbn == 1);

  assert(ctxlay->n == 0);
  // const Partrait &inpic = pic0;
  Partrait inpic = pic0;
  assert(inpic.w * inpic.h * 3 == ctrlay->n);

  double *rgb = new double[ctrlay->n];
  btodv(inpic.rgb, rgb, ctrlay->n);
  
  encude(rgb, ctrlay->n, cugenin);
  delete[] rgb;

  double *cugenout;
  cugenout = (double *)gen->feed(cugenin, NULL);

  Partrait tgtpic = pic0;
  assert(tgtlay->n == tgtpic.w * tgtpic.h * 3);
  rgb = new double[tgtlay->n];
  btodv(tgtpic.rgb, rgb, tgtlay->n);
  double *curgb;
  cumake(&curgb, tgtlay->n);
  encude(rgb, tgtlay->n, curgb);
  cuaddvec(curgb, cugenout, tgtlay->n, cugenout);
  cufree(curgb);

  if (outpic) {
    decude(cugenout, tgtlay->n, rgb);
    outpic->create(tgtpic.w, tgtpic.h);
    dtobv(rgb, outpic->rgb, tgtlay->n);
  }

  delete[] rgb;
}

void Zoomgen::burn(double pi, Zoomdis *dis, const class Partrait *pic, double ganlev) {
  if (dis) {
    assert(dis->dis->inn == tgtlay->n);

    double sc = dis->score(this);

//    if (sc > 0.5)
//      return;
//dis->burn(0.5, 0.0);


//  dis->burn(1.0, 0.0);

  dis->burn(sc + 1.0, 0.0);


  }

  if (ganlev > 0) {
    assert(tgtlay->n == pic->w * pic->h * 3);
    btodv(pic->rgb, tgtbuf, pic->w * pic->h * 3);
    encude(tgtbuf, tgtlay->n, cugentgt);

    cusubvec(cugentgt, gen->output(), tgtlay->n, cugentgt);
    cumuld(cugentgt, ganlev, tgtlay->n, cugentgt);
    cuaddvec(cugentgt, gen->foutput(), tgtlay->n, gen->foutput());
  }

  gen->update_stats();
  gen->train(pi);
}

void Zoomgen::burn(double pi, const Partrait &pic) {
  assert(pic.w * pic.h * 3 == tgtlay->n);

  double *rgb = new double[tgtlay->n];
  btodv(pic.rgb, rgb, tgtlay->n);
  encude(rgb, tgtlay->n, cugentgt);
  delete[] rgb;

  gen->target(cugentgt);
  gen->update_stats();
  gen->train(pi);
}

}

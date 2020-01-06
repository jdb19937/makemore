#define __MAKEMORE_SUPRGEN_CC__ 1

#include <string>
#include <algorithm>

#include "cudamem.hh"
#include "tron.hh"
#include "multitron.hh"
#include "encoder.hh"
#include "supergen.hh"
#include "parson.hh"
#include "strutils.hh"
#include "imgutils.hh"
#include "cholo.hh"
#include "numutils.hh"
#include "superdis.hh"

namespace makemore {

using namespace std;

Supergen::Supergen(const std::string &_dir, unsigned int _mbn) : Project(_dir, _mbn) {
  assert(mbn > 0);

  assert(config["type"] == "supergen");
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

Supergen::~Supergen() {
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


void Supergen::report(const char *prog) {
  fprintf(
    stderr,
    "%s %s rounds=%u gen_err2=%g gen_errm=%g\n",
    prog, dir.c_str(), rounds, gen->err2, gen->errm
  );
}

void Supergen::save() {
  genmap->save();
}

void Supergen::load() {
  genmap->load();
}

void Supergen::generate(const double *ctr, class Partrait *prt, class Styler *sty, bool bp) {
  assert(mbn == 1);

  encude(ctr, ctrlay->n, cugenin + ctxlay->n);

  const double *cugenout;
  if (bp) {
    cuzero(cugenfin, gen->inn);
    cugenout = gen->feed(cugenin, cugenfin);
  } else {
    cugenout = gen->feed(cugenin, NULL);
  }

  if (prt) {
    decude(cugenout, gen->outn, tgtbuf);

    prt->set_pose(Pose::STANDARD);

    assert(!is_rgb);
    if (tgtlay->n == prt->w * prt->h * 4) {
      if (!prt->alpha)
        prt->alpha = new uint8_t[prt->w * prt->h];
      labargba(tgtbuf, prt->w * prt->h, prt->rgb, prt->alpha);
    } else if (tgtlay->n == prt->w * prt->h * 3) {
      if (prt->alpha) {
        delete[] prt->alpha;
        prt->alpha = NULL;
      }
      labrgb(tgtbuf, prt->w * prt->h * 3, prt->rgb);
    } else {
      assert(0);
    }

#if 0
    Partrait stdprt(256, 256);
    stdprt.set_pose(Pose::STANDARD);
    if (tgtlay->n == 256 * 256 * 4)
      stdprt.alpha = new uint8_t[256 * 256];

    assert(!is_rgb);
    if (tgtlay->n == 256 * 256 * 4)
      labargba(tgtbuf, 256 * 256, stdprt.rgb, stdprt.alpha);
    else
      labrgb(tgtbuf, 256 * 256 * 3, stdprt.rgb);

    prt->create(256, 256);
    if (ctxtype == 2)
      prt->fill_gray();
    else
      prt->fill_white();

    Pose pose = Pose::STANDARD;

    prt->set_pose(pose);

    if (stdprt.alpha && !prt->alpha)
      prt->alpha = new uint8_t[256 * 256];
    stdprt.warp(prt);
#endif
  }
}

void Supergen::burn(const class Partrait &prt, double pi) {
  assert(prt.w * prt.h * 3 == tgtlay->n);

#if 0
  if (prt.alpha)
    rgbalaba(prt.rgb, prt.alpha, prt.w * prt.h, tgtbuf);
  else
    rgblaba(prt.rgb, prt.w * prt.h, tgtbuf);
#endif
  rgblab(prt.rgb, prt.w * prt.h * 3, tgtbuf);


  encude(tgtbuf, tgtlay->n, cugentgt);
  gen->target(cugentgt);

#if 0
  cudalpha(gen->foutput(), cugentgt, prt.w * prt.h);
#endif

  if (focus)
    cufocus(gen->foutput(), cutgtlayx, cutgtlayy, tgtlay->n);

  gen->update_stats();
  gen->train(pi);
}

void Supergen::burn(const class Partrait &prt, double pi, Superdis *dis) {
  assert(dis->inplay->n == tgtlay->n);
  assert(dis->dis->inn == tgtlay->n);
  assert(prt.w * prt.h * 3 == tgtlay->n);

  double sc = dis->score(this);
//fprintf(stderr, "sc=%lf\n", sc);
  dis->burn(1.0, 0.0);

  double reinf = 2e-2;
//  double reinf = 2e-4;
  {
    rgblab(prt.rgb, 256 * 256 * 3, tgtbuf);
    encude(tgtbuf, tgtlay->n, cugentgt);

    cusubvec(cugentgt, gen->output(), tgtlay->n, cugentgt);
    cumuld(cugentgt, reinf, tgtlay->n, cugentgt);
    cuaddvec(cugentgt, gen->foutput(), tgtlay->n, gen->foutput());
  }

#if 0
  double *ganbuf = new double[gen->outn];
  double *outbuf = new double[gen->outn];

  decude(gen->output(), gen->outn, outbuf);
  decude(gen->foutput(), gen->outn, ganbuf);
  gen->update_stats();
  cuzero(gen->foutput(), gen->outn);

  assert(gen->outn == prt.w * prt.h * 3);
  rgblab(prt.rgb, prt.w * prt.h * 3, tgtbuf);

  for (unsigned int j = 0; j < gen->outn; ++j) {
   // tgtbuf[j] = outbuf[j] + (tgtbuf[j] - outbuf[j]) * fabs(ganbuf[j]);
//    tgtbuf[j] = outbuf[j] + ganbuf[j];

//     tgtbuf[j] = outbuf[j] + 1e-2 * (tgtbuf[j] - outbuf[j]) + ganbuf[j];
//     tgtbuf[j] = outbuf[j] + ganbuf[j] * (1.0 + 1e0 * ganbuf[j] * (tgtbuf[j] - outbuf[j]));

     double sgn = (ganbuf[j] >= 0 ? 1.0 : -1.0);
     double del = tgtbuf[j] - outbuf[j];

//     double kk = 0.1;
//     tgtbuf[j] = outbuf[j] + ganbuf[j] * (1.0 + kk * sgn * del);

    double rr = 1e-3;

     tgtbuf[j] = outbuf[j] + ganbuf[j] + rr * del;
  }

  delete[] ganbuf;
  delete[] outbuf;

  encude(tgtbuf, tgtlay->n, cugentgt);
  gen->target(cugentgt);
#endif

  gen->update_stats();

  gen->train(pi);
}

#if 0
void Supergen::burn(const class Partrait &prt, double pi, Superdis *dis) {
  assert(dis->inplay->n == tgtlay->n);
  assert(dis->cls->inn == tgtlay->n);
  assert(prt.w * prt.h * 3 == tgtlay->n);

  double *lab, *cuclsbuf;
  cumake(&cuclsbuf, dis->cls->outn);
  lab = new double[dis->cls->inn];

  rgblab(prt.rgb, dis->cls->inn, lab);
  double d = 0.5 / 256.0;
  for (unsigned int j = 0; j < dis->cls->inn; ++j)
    lab[j] += randrange(-d, d);

  encude(lab, dis->cls->inn, dis->cuclsin);
  dis->cls->feed(dis->cuclsin);
  cucopy(dis->cls->output(), dis->cls->outn, cuclsbuf);

  const double *disout = dis->dis->feed(cuclsbuf, NULL);
  double sc1;
  decude(disout, 1, &sc1);

  const double *clsout = dis->cls->feed(gen->output(), gen->foutput());
  disout = dis->dis->feed(clsout, NULL);
  double sc0;
  decude(disout, 1, &sc0);

fprintf(stderr, "sc0=%lf, sc1=%lf, dsc=%lf\n", sc0, sc1, sc1 - sc0);
//  if (sc1 - sc0 > 0.0) {
//  if (sc0 < 0 && sc1 > 0) {
{
    dis->cls->target(cuclsbuf);
    dis->cls->update_stats();
    dis->cls->train(0);

    cufree(cuclsbuf);
    delete[] lab;

    gen->update_stats();
    gen->train(pi);
  }
}
#endif

#if 0
void Supergen::burn(Superdis *dis, double pi, const Partrait *rprt, double reinf) {
  assert(dis->inplay->n == tgtlay->n);

  (void) dis->score(this);
  dis->burn(1.0, 0.0);

#if 0
  cudalpha(gen->foutput(), cugentgt, prt.w * prt.h);
#endif

  if (focus)
    cufocus(gen->foutput(), cutgtlayx, cutgtlayy, tgtlay->n);
  if (rprt && reinf > 0) {
    rgblab(rprt->rgb, 256 * 256 * 3, tgtbuf);
    encude(tgtbuf, tgtlay->n, cugentgt);

    cusubvec(cugentgt, gen->output(), tgtlay->n, cugentgt);
    cumuld(cugentgt, reinf, tgtlay->n, cugentgt);
    cuaddvec(cugentgt, gen->foutput(), tgtlay->n, gen->foutput());
  }

  gen->update_stats();
  gen->train(pi);
}
#endif

}

#define __MAKEMORE_GENERATOR_CC__ 1

#include <string>
#include <algorithm>

#include "cudamem.hh"
#include "tron.hh"
#include "multitron.hh"
#include "encoder.hh"
#include "generator.hh"
#include "parson.hh"
#include "strutils.hh"
#include "imgutils.hh"
#include "cholo.hh"
#include "numutils.hh"

namespace makemore {

using namespace std;

Generator::Generator(const std::string &_dir, unsigned int _mbn) : Project(_dir, _mbn) {
  assert(mbn > 0);

  assert(config["type"] == "generator");
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

  char genmapfn[4096], gentopfn[4096];
  sprintf(gentopfn, "%s/gen.top", dir.c_str());
  sprintf(genmapfn, "%s/gen.map", dir.c_str());
  gentop = new Topology;
  gentop->load_file(gentopfn);
  genmap = new Mapfile(genmapfn);
  gen = new Multitron(*gentop, genmap, mbn, false);

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

Generator::~Generator() {
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


void Generator::report(const char *prog) {
  fprintf(
    stderr,
    "%s %s rounds=%u gen_err2=%g gen_errm=%g\n",
    prog, dir.c_str(), rounds, gen->err2, gen->errm
  );
}

void Generator::save() {
  genmap->save();
}

void Generator::load() {
  genmap->load();
}

void Generator::generate(const Parson &prs, class Partrait *prt, class Styler *sty, bool bp) {
  assert(mbn == 1);

  switch (ctxtype) {
  case 0:
    {
      assert(ctxlay->n >= 192);
      memcpy(ctxbuf, prs.sketch, 192 * sizeof(double));

      Hashbag hb;
      prs.bagtags(&hb);
      assert(Hashbag::n >= 64);
      assert(ctxlay->n >= 192 + 64);
      memcpy(ctxbuf + 192, hb.vec, 64 * sizeof(double));

      assert(ctxlay->n >= 192 + 64 + 3);
      ctxbuf[256] = prs.angle;
      ctxbuf[257] = prs.stretch;
      ctxbuf[258] = prs.skew;

      memset(ctxbuf + 259, 0, (ctxlay->n - 259) * sizeof(double));
    }
    break;
  case 2:
    {
      double sex = 0.5;
      if (prs.has_tag("male"))
        sex += 0.5;
      if (prs.has_tag("female"))
        sex -= 0.5;

      double age = 0.5;
      if (prs.has_tag("old"))
        age += 0.5;
      if (prs.has_tag("young"))
        age -= 0.5;

      int race[4] = {0, 0, 0, 0};
      if (prs.has_tag("white"))
        race[0] += 1.0;
      if (prs.has_tag("black"))
        race[1] += 1.0;
      if (prs.has_tag("asian"))
        race[2] += 1.0;
      if (prs.has_tag("hispanic"))
        race[3] += 1.0;

      assert(ctxlay->n >= 9);
      memset(ctxbuf + 9, 0, (ctxlay->n - 9) * sizeof(double));

      ctxbuf[0] = sex;
      ctxbuf[1] = age;
      ctxbuf[2] = race[0];
      ctxbuf[3] = race[1];
      ctxbuf[4] = race[2];
      ctxbuf[5] = race[3];
      ctxbuf[6] = prs.angle;
      ctxbuf[7] = prs.stretch;
      ctxbuf[8] = prs.skew;
    }
    break;
  case 3:
    {
      double sex = 0.5;
      if (prs.has_tag("male"))
        sex += 0.5;
      if (prs.has_tag("female"))
        sex -= 0.5;

      double age = 0.5;
      if (prs.has_tag("old"))
        age += 0.5;
      if (prs.has_tag("young"))
        age -= 0.5;

      int race[4] = {0, 0, 0, 0};
      if (prs.has_tag("white"))
        race[0] += 1.0;
      if (prs.has_tag("black"))
        race[1] += 1.0;
      if (prs.has_tag("asian"))
        race[2] += 1.0;
      if (prs.has_tag("hispanic"))
        race[3] += 1.0;

      assert(ctxlay->n >= 192 + 9);
      memcpy(ctxbuf, prs.sketch, 192 * sizeof(double));

      ctxbuf[192+0] = sex;
      ctxbuf[192+1] = age;
      ctxbuf[192+2] = race[0];
      ctxbuf[192+3] = race[1];
      ctxbuf[192+4] = race[2];
      ctxbuf[192+5] = race[3];
      ctxbuf[192+6] = prs.angle;
      ctxbuf[192+7] = prs.stretch;
      ctxbuf[192+8] = prs.skew;
    }
    break;
  default:
    assert(0);
  }
  
  encude(ctxbuf, ctxlay->n, cugenin);

  assert(Parson::ncontrols == ctrlay->n);
  if (sty) {
    assert(sty->dim == ctrlay->n);
    sty->generate(prs, ctrbuf, 2.0);
    encude(ctrbuf, ctrlay->n, cugenin + ctxlay->n);
  } else {
    encude(prs.controls, ctrlay->n, cugenin + ctxlay->n);
  }

  const double *cugenout;
  if (bp) {
    cuzero(cugenfin, gen->inn);
    cugenout = gen->feed(cugenin, cugenfin);
  } else {
    cugenout = gen->feed(cugenin, NULL);
  }

  if (prt) {
    decude(cugenout, gen->outn, tgtbuf);
    assert(tgtlay->n == 256 * 256 * 3);

    Partrait stdprt(256, 256);
    stdprt.set_pose(Pose::STANDARD);
    if (is_rgb)
      dtobv(tgtbuf, stdprt.rgb, 256 * 256 * 3);
    else
      labrgb(tgtbuf, 256 * 256 * 3, stdprt.rgb);

    prt->create(256, 256);
    if (ctxtype == 2)
      prt->fill_gray();
    else
      prt->fill_white();

    Pose pose = Pose::STANDARD;
    pose.angle = prs.angle;
    pose.stretch = prs.stretch;
    pose.skew = prs.skew;

    prt->set_pose(pose);
    stdprt.warpover(prt);

    for (unsigned int j = 0; j < Parson::ntags; ++j) {
      if (!*prs.tags[j])
        break;
      prt->tags.push_back(prs.tags[j]);
    }
  }
}

void Generator::burn(const class Partrait &prt, double pi) {
  assert(prt.w * prt.h * 4 == tgtlay->n);
  assert(prt.alpha);

  rgblab(prt.rgb, prt.w * prt.h * 3, buf);
  {
    const double *u = buf;
    const uint8_t *a = prt.alpha;
    double *t = tgtbuf;
    for (unsigned int j = 0, jn = prt.w * prt.h; j < jn; ++j) {
      double ax = (double)*a++ / 255.0;
      *t++ = *u++ * ax;
      *t++ = *u++ * ax;
      *t++ = *u++ * ax;
      *t++ = ax;
    }
  }
  encude(tgtbuf, tgtlay->n, cugentgt);
  gen->target(cugentgt, false);

  if (focus)
    cufocus(gen->foutput(), cutgtlayx, cutgtlayy, tgtlay->n);

  gen->update_stats();
  gen->train(pi);
}

}

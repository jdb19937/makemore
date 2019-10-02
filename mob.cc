#define __MAKEMORE_MOB_CC__ 1
#include "mob.hh"

#include "ppm.hh"

namespace makemore {

void make_mob(Supergen *gen, Styler *sty, Automasker *am, Partrait *prt) {
  unsigned long seed = 0;

  double dev = 1.0;

  unsigned int w = 256;
  unsigned int h = 256;

  PPM out(1024, 1024);
  seedrand(seed);

  // assert(egd.ctrlay->n == 512);

  std::multimap<int, Partrait *> ymap;
  std::map<Partrait *, int > xmap;

  unsigned int nc = 0;
  while (nc < 36) {
#if 0
    unsigned int y = 160 * (int)(nc / 8) + (randuint() % 96) - 96;
    int x = 160 * (nc % 8) + (randuint() % 96) - 96;
#endif

    unsigned int y = 160 * (int)(nc / 6) + (randuint() % 96) - 96;
    int x = 160 * (nc % 6) + (randuint() % 96) - 96;

    double scale = 64.0;

    Partrait *npar = new Partrait(256, 256);
    npar->fill_black();
    npar->set_pose(Pose::STANDARD);

#if 1
    Parson prs;


memset(prs.tags, 0, sizeof(prs.tags));
//prs.add_tag("female");
//prs.add_tag("white");
//prs.add_tag("young");

    for (unsigned int j = 0; j < Parson::ncontrols; ++j)
      prs.controls[j] = randgauss();
    prs.angle = randrange(-0.1, 0.1);
    prs.stretch = 1.0 + randrange(-0.1, 0.1);
    prs.skew = randrange(-0.1, 0.1);

//    gen->generate(prs, npar, sty);
    sty->tag_cholo["base"]->generate(prs.controls, prs.controls);
    gen->generate(prs.controls, npar);

#else
  Parson &prs(*gen->zone->pick());
    gen->generate(prs, npar);
#endif

    Pose pose = Pose::STANDARD;
    pose.angle = prs.angle;
    pose.stretch = prs.stretch;
    pose.skew = prs.skew;
    pose.scale = scale;

   Partrait *spar = new Partrait(256, 256);
   spar->set_pose(pose);
   spar->fill_gray();
   npar->warp(spar);


    ymap.insert(std::make_pair(y, spar));
    xmap[spar] = x;
    ++nc;
  }

  for (auto yi : ymap) {
    int y = (int)yi.first;
    Partrait *spar = yi.second;
    int x = xmap[spar];

fprintf(stderr, "y=%d\n", y);

if (am)
  am->automask(spar);


if (spar->alpha) {
#if 0
for (unsigned int j = 0; j < w * h; ++j) {
  double x = spar->alpha[j];
  x -= 0.4;
  x /= 0.2;
  if (x > 255) x = 255;
  if (x < 0) x = 0;
  spar->alpha[j] = x;
}
#endif

    out.pastealpha(spar->rgb, spar->alpha, w, h, x, y);
} else
    out.paste(spar->rgb, w, h, x, y);
  }

  prt->clear();
  prt->w = out.w;
  prt->h = out.h;
  prt->rgb = out.data;
  out.data = NULL;
  prt->alpha = NULL;
}

}

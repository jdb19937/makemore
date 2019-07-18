#define __MAKEMORE_AUTOGAZER_CC__ 1

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
#include "autogazer.hh"
#include "partrait.hh"

namespace makemore {

using namespace std;

Autogazer::Autogazer(const std::string &_dir) : Project(_dir, 1) {
  assert(mbn == 1);
  assert(config["type"] == "autogazer");

  char gazoutlayfn[4096];
  sprintf(gazoutlayfn, "%s/gazoutput.lay", dir.c_str());
  gazoutlay = new Layout;
  gazoutlay->load_file(gazoutlayfn);

  char gazinlayfn[4096];
  sprintf(gazinlayfn, "%s/gazinput.lay", dir.c_str());
  gazinlay = new Layout;
  gazinlay->load_file(gazinlayfn);

  char gazmapfn[4096], gaztopfn[4096];
  sprintf(gaztopfn, "%s/gaz.top", dir.c_str());
  sprintf(gazmapfn, "%s/gaz.map", dir.c_str());
  gaztop = new Topology;
  gaztop->load_file(gaztopfn);
  gazmap = new Mapfile(gazmapfn);
  gaz = new Multitron(*gaztop, gazmap, mbn, false);

  assert(gaz->outn == gazoutlay->n);
  assert(gazoutlay->n == 2);

  cumake(&cugaztgt, gaz->outn);
  cumake(&cugazin, gaz->inn);

  rounds = 0;
}

Autogazer::~Autogazer() {
  delete gaz;
  delete gazmap;
  delete gaztop;

  delete gazinlay;
  delete gazoutlay;

  cufree(cugazin);
  cufree(cugaztgt);
}



void Autogazer::report(const char *prog) {
  fprintf(
    stderr,
    "%s %s rounds=%u\n"
    "%s %s gaz_err2=%g gaz_errm=%g\n"
    "\n",
    prog, dir.c_str(), rounds,
    prog, dir.c_str(), gaz->err2, gaz->errm
  );
}

void Autogazer::save() {
  gazmap->save();
}

void Autogazer::load() {
  gazmap->load();
}

void Autogazer::observe(const Partrait &par, double mu) {
  assert(par.has_mark());

  assert(par.has_gaze());
  Point gz = par.get_gaze();

  Triangle mark;
  mark.p = Point(64, 64);
  mark.q = Point(64 + 256, 64);
  mark.r = Point(64 + 128, 64 + 128 * 3);

  Partrait leye(128, 128);
  leye.set_mark(mark);
  par.warp(&leye);

  mark.p = Point(64 - 256, 64);
  mark.q = Point(64, 64);
  mark.r = Point(64 - 128, 64 + 128 * 3);

  Partrait reye(128, 128);
  reye.set_mark(mark);
  par.warp(&reye);

  assert(gaz->inn ==
    leye.w * leye.h * 3 +
    reye.w * reye.h * 3 +
    12
  );

  leye.encudub(cugazin);
  reye.encudub(cugazin + 128 * 128 * 3);

  mark = par.get_mark();
  double ext[12];
  ext[0] = mark.p.x / (double)par.w;
  ext[1] = mark.p.y / (double)par.h;
  ext[2] = mark.q.x / (double)par.w;
  ext[3] = mark.q.y / (double)par.h;
  ext[4] = mark.r.x / (double)par.w;
  ext[5] = mark.r.y / (double)par.h;

  Pose pose = par.get_pose();
  ext[6] = pose.center.x / (double)par.w;
  ext[7] = pose.center.y / (double)par.h;
  ext[8] = pose.scale / (double)par.h;
  ext[9] = pose.angle;
  ext[10] = pose.stretch;
  ext[11] = pose.skew;
  
  encude(ext, 12, cugazin + 2 * (128 * 128 * 3));

  gaz->feed(cugazin, NULL);

  assert(gaz->outn == 2);

  double tgt[2] = {gz.x, gz.y};
//fprintf(stderr, "tgt=%lf,%lf\n", tgt[0], tgt[1]);
  encude(tgt, 2, cugaztgt);

  gaz->target(cugaztgt);
  gaz->train(mu);
}

void Autogazer::autogaze(Partrait *parp) {
  const Partrait &par(*parp);

  assert(par.has_mark());

  Triangle mark;
  mark.p = Point(64, 64);
  mark.q = Point(64 + 256, 64);
  mark.r = Point(64 + 128, 64 + 128 * 3);

  Partrait leye(128, 128);
  leye.set_mark(mark);
  par.warp(&leye);

  mark.p = Point(64 - 256, 64);
  mark.q = Point(64, 64);
  mark.r = Point(64 - 128, 64 + 128 * 3);

  Partrait reye(128, 128);
  reye.set_mark(mark);
  par.warp(&reye);

  assert(gaz->inn ==
    leye.w * leye.h * 3 +
    reye.w * reye.h * 3 +
    12
  );

  leye.encudub(cugazin);
  reye.encudub(cugazin + 128 * 128 * 3);

  mark = par.get_mark();
  double ext[12];
  ext[0] = mark.p.x / (double)par.w;
  ext[1] = mark.p.y / (double)par.h;
  ext[2] = mark.q.x / (double)par.w;
  ext[3] = mark.q.y / (double)par.h;
  ext[4] = mark.r.x / (double)par.w;
  ext[5] = mark.r.y / (double)par.h;

  Pose pose = par.get_pose();
  ext[6] = pose.center.x / (double)par.w;
  ext[7] = pose.center.y / (double)par.h;
  ext[8] = pose.scale / (double)par.h;
  ext[9] = pose.angle;
  ext[10] = pose.stretch;
  ext[11] = pose.skew;
  
  encude(ext, 12, cugazin + 2 * (128 * 128 * 3));

  const double *cugazout = gaz->feed(cugazin, NULL);

  double gz[2];
  decude(cugazout, 2, gz);
//fprintf(stderr, "gz0=%lf,%lf\n", gz[0], gz[1]);
  parp->set_gaze(Point(gz[0], gz[1]));
}

}

#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>

#include "urb.hh"
#include "zone.hh"
#include "cudamem.hh"
#include "numutils.hh"
#include "imgutils.hh"
#include "cholo.hh"
#include "strutils.hh"
#include "partrait.hh"
#include "catalog.hh"
#include "generator.hh"
#include "automasker.hh"


using namespace makemore;
using namespace std;

std::string pathfn(const std::string &path) {
  const char *p = path.c_str();
  const char *q = strrchr(p, '/');
  if (!q)
    return path;
  return std::string(q + 1);
}

int main(int argc, char **argv) {
  seedrand();

  unsigned int mbn = 1;
  unsigned int w = 256, h = 256;

  assert(argc == 3);
  Zone zone(argv[1]);
  Automasker *am = new Automasker(argv[2]);

fprintf(stderr, "starting\n");

Partrait *par = new Partrait;
//  Partrait *spar = new Partrait[1024];

  int i = 0;
  while (1) {
    unsigned int which = randuint() % zone.n;

    Parson *prs = zone.db + which;

    std::string fn = prs->srcfn;

    assert(fn.length());

    Partrait prt;
    prt.load(fn);
//fprintf(stderr, "read %s\n", fn.c_str());

    assert(prt.w == w);
    assert(prt.h == h);
    assert(prt.alpha);

    Pose adjpose = prt.get_pose();
    adjpose.stretch += randrange(-0.03, 0.03);
    adjpose.skew += randrange(-0.03, 0.03);
    adjpose.angle += randrange(-0.05, 0.05);
    adjpose.scale += randrange(-4.0, 4.0);
    adjpose.center.x += randrange(-8.0, 8.0);
    adjpose.center.y += randrange(-8.0, 8.0);

    Partrait adjprt(w, h);
    adjprt.fill_black();
    adjprt.alpha = new uint8_t[w * h]();
    adjprt.set_pose(adjpose);
    prt.warp(&adjprt);

    if (randuint() % 2) {
      adjprt.reflect();
    }

    am->observe(adjprt, 0.00001);

    if (i % 100 == 0) {
      am->report("burnmask");
      am->save();
    }

    ++i;
  }
}


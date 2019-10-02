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
#include "autoposer.hh"


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
  Autoposer *ap = new Autoposer(argv[2]);

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

    Triangle mark = prt.get_mark();
    double de = 8.0;
    double dm = 16.0;
    mark.p.x += randrange(-de, de);
    mark.p.y += randrange(-de, de);
    mark.q.x += randrange(-de, de);
    mark.q.y += randrange(-de, de);
    mark.r.x += randrange(-dm, dm);
    mark.r.y += randrange(-dm, dm);
    Pose adjpose = Pose(mark);

    Partrait adjprt(w, h);
    adjprt.set_pose(adjpose);
    prt.warp(&adjprt);

    if (randuint() % 2) {
      adjprt.reflect();
    }

    ap->observe(adjprt, 0.0000001);

    if (i % 100 == 0) {
      ap->report("burnseg");
      ap->save();
    }

    ++i;
  }
}


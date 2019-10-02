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
#include "autogazer.hh"


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

  assert(argc == 3);
  Catalog cat(argv[1]);
  Autogazer *ag = new Autogazer(argv[2]);

fprintf(stderr, "starting\n");

Partrait *par = new Partrait;
//  Partrait *spar = new Partrait[1024];

  int i = 0;
  while (1) {
    Partrait prt;
    cat.pick(&prt);

    assert(prt.has_gaze());
    if (randuint() % 2) {
      prt.reflect();
    }

 
    assert(prt.has_mark());
    {
      Triangle mark = prt.get_mark();
      double d = 1.0;
      mark.p.x += randrange(-d, d);
      mark.p.y += randrange(-d, d);
      mark.q.x += randrange(-d, d);
      mark.q.y += randrange(-d, d);
      mark.r.x += randrange(-d, d);
      mark.r.y += randrange(-d, d);
      prt.set_mark(mark);
    }

    ag->observe(prt, 0.00001);

    ++i;
    if (i % 100 == 0) {
      ag->report("burngaze");
      ag->save();
    }
  }
}


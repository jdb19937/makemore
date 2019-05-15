#define __MAKEMORE_AUTOPOSER_CC__ 1

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
#include "autoposer.hh"
#include "partrait.hh"

namespace makemore {

using namespace std;

static string read_word(FILE *fp, char sep) {
  int c = getc(fp);
  if (c == EOF)
    return "";

  char buf[2];
  buf[0] = (char)c;
  buf[1] = 0;
  string word(buf);

  while (1) {
    c = getc(fp);
    if (c == EOF)
      return "";
    if (c == sep)
      break;
    buf[0] = (char)c;
    word += buf;
  }

  return word;
}

Autoposer::Autoposer(const std::string &_dir) : Project(_dir, 1) {
  assert(mbn == 1);
  assert(config["type"] == "autoposer");

  char segoutlayfn[4096];
  sprintf(segoutlayfn, "%s/segoutput.lay", dir.c_str());
  segoutlay = new Layout;
  segoutlay->load_file(segoutlayfn);

  char seginlayfn[4096];
  sprintf(seginlayfn, "%s/seginput.lay", dir.c_str());
  seginlay = new Layout;
  seginlay->load_file(seginlayfn);

  char segmapfn[4096], segtopfn[4096];
  sprintf(segtopfn, "%s/seg.top", dir.c_str());
  sprintf(segmapfn, "%s/seg.map", dir.c_str());
  segtop = new Topology;
  segtop->load_file(segtopfn);
  segmap = new Mapfile(segmapfn);
  seg = new Multitron(*segtop, segmap, mbn, false);

  assert(seg->outn == segoutlay->n);
  assert(segoutlay->n == 6);

  cumake(&cusegtgt, seg->outn);
  cumake(&cusegin, seg->inn);

  rounds = 0;
}

Autoposer::~Autoposer() {
  delete seg;
  delete segmap;
  delete segtop;

  delete seginlay;
  delete segoutlay;

  cufree(cusegin);
  cufree(cusegtgt);
}



void Autoposer::report(const char *prog) {
  fprintf(
    stderr,
    "%s %s rounds=%u\n"
    "%s %s seg_err2=%g seg_errm=%g\n"
    "\n",
    prog, dir.c_str(), rounds,
    prog, dir.c_str(), seg->err2, seg->errm
  );
}

void Autoposer::save() {
  segmap->save();
}

void Autoposer::load() {
  segmap->load();
}

void Autoposer::observe(const Partrait &par, double mu) {
  assert(par.has_mark());
  Triangle mark = par.get_mark();

  assert(seg->inn == par.w * par.h * 3);
  par.encudub(cusegin);

  seg->feed(cusegin, NULL);

  mark.p.x /= (double)par.w;
  mark.p.y /= (double)par.h;
  mark.q.x /= (double)par.w;
  mark.q.y /= (double)par.h;
  mark.r.x /= (double)par.w;
  mark.r.y /= (double)par.h;

  assert(seg->outn == 6);
  assert(sizeof(mark) == 6 * sizeof(double));
  encude((double *)&mark, 6, cusegtgt);

  seg->target(cusegtgt);
  seg->train(mu);
}

void Autoposer::autopose(Partrait *parp) {
  assert(parp->has_mark());
  assert(seginlay->n == 256 * 256 * 3);
  Partrait x(256, 256);
  x.set_pose(Pose::STANDARD);
  parp->warp(&x);

  x.encudub(cusegin);

  const double *cusegout = seg->feed(cusegin, NULL);

  Triangle xmark;
  decude(cusegout, 6, (double *)&xmark);

  xmark.p.x *= 256.0;
  xmark.p.y *= 256.0;
  xmark.q.x *= 256.0;
  xmark.q.y *= 256.0;
  xmark.r.x *= 256.0;
  xmark.r.y *= 256.0;

fprintf(stderr, "xmark=(%lf,%lf) (%lf,%lf) (%lf,%lf)\n", xmark.p.x, xmark.p.y, xmark.q.x, xmark.q.y, xmark.r.x, xmark.r.y);

  Triangle pmark = parp->get_mark();
  Triangle qmark = x.get_mark();
  xmark.p = makemore::trimap(xmark.p, qmark, pmark);
  xmark.q = makemore::trimap(xmark.q, qmark, pmark);
  xmark.r = makemore::trimap(xmark.r, qmark, pmark);

  parp->set_mark(xmark);
}

}

#include <stdio.h>
#include <dirent.h>
#include <stdlib.h>

#include "improver.hh"
#include "urb.hh"
#include "zone.hh"
#include "cudamem.hh"
#include "numutils.hh"
#include "imgutils.hh"
#include "strutils.hh"
#include "encgendis.hh"
#include "cholo.hh"
#include "warp.hh"
#include "triangle.hh"
#include "partrait.hh"

using namespace makemore;
using namespace std;

int main() {
  seedrand();

  unsigned int mbn = 1;
  Encgendis egd("oldseg.proj", mbn);

  std::vector<std::string> srcimages;

  std::string srcdir = "/home/dan/makemore/allmugs";
  struct dirent *de;
  DIR *dp = opendir(srcdir.c_str());
  assert(dp);
  while ((de = readdir(dp))) {
    if (*de->d_name == '.')
      continue;
    srcimages.push_back(srcdir + "/" + de->d_name);
  }
  closedir(dp);
  std::sort(srcimages.begin(), srcimages.end());
  assert(srcimages.size());


fprintf(stderr, "starting\n");

  int i = 0;
  while (1) {
    unsigned int k = randuint() % srcimages.size();
    string imagefn = srcimages[k];
    Partrait srcpar;
    srcpar.load(imagefn);
    if (!srcpar.has_mark())
      continue;

    Partrait par = srcpar;
    Triangle mark = par.get_mark();

    assert(egd.seg->inn == par.w * par.h * 3);
    par.encudub(egd.cusegin);
    const double *cusegout = egd.seg->feed(egd.cusegin, NULL);
    decude(cusegout, egd.seg->outn, egd.segbuf);

    egd.segbuf[0] = mark.p.x / (double)par.w;
    egd.segbuf[1] = mark.p.y / (double)par.h;
    egd.segbuf[2] = mark.q.x / (double)par.w;
    egd.segbuf[3] = mark.q.y / (double)par.h;
    egd.segbuf[4] = mark.r.x / (double)par.w;
    egd.segbuf[5] = mark.r.y / (double)par.h;

     assert(egd.segoutlay->n == 6);
     assert(egd.seg->outn == 6);

    encude(egd.segbuf, 6, egd.cusegtgt);
    egd.seg->target(egd.cusegtgt);
    egd.seg->train(0.000005);
    
    if (i % 100 == 0) {
      egd.report("burnseg");
fprintf(stderr, "saving\n");
      egd.save();
fprintf(stderr, "saved\n");
    }
    ++i;
  }
}


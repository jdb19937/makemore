#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>

#include "urb.hh"
#include "zone.hh"
#include "cudamem.hh"
#include "numutils.hh"
#include "imgutils.hh"
#include "encgendis.hh"
#include "cholo.hh"
#include "strutils.hh"
#include "partrait.hh"


using namespace makemore;
using namespace std;

int main() {
  seedrand();

  unsigned int mbn = 1;
  Encgendis egd("big.proj", mbn);
  double *tmpd = new double[1<<20];
  unsigned int w = 256, h = 256;

  std::vector<std::string> srcimages;

  std::string srcdir = "/spin/dan/celeba.aligned";
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

  assert(egd.tgtlay->n == w * h * 3);

fprintf(stderr, "starting\n");

  int i = 0;
  while (1) {
    memset(egd.ctxbuf, 0, mbn * egd.ctxlay->n * sizeof(double));

    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
again:
      unsigned int which = randuint() % srcimages.size();
      std::string srcfn = srcimages[which];

      Partrait par;
      par.load(srcfn);

      if (randuint() % 2)
        par.reflect();

      assert(par.w * par.h * 3 == egd.tgtlay->n);
      btodv(par.rgb, egd.tgtbuf + mbi * egd.tgtlay->n, egd.tgtlay->n);

      double *ctx = egd.ctxbuf + mbi * egd.ctxlay->n;
      if (par.has_tag("male")) { ctx[0] = 1.0; }
      if (par.has_tag("female")) { ctx[1] = 1.0; }
      if (par.has_tag("white")) { ctx[2] = 1.0; }
      if (par.has_tag("black")) { ctx[3] = 1.0; }
      if (par.has_tag("hispanic")) { ctx[4] = 1.0; }
      if (par.has_tag("asian")) { ctx[5] = 1.0; }
      if (par.has_tag("glasses")) { ctx[6] = 1.0; }
    }

    egd.burn(0.00002, 0.00002);

    if (i % 100 == 0) {
      egd.report("burnbig");
fprintf(stderr, "saving\n");
      egd.save();
fprintf(stderr, "saved\n");
    }
    ++i;
  }
}


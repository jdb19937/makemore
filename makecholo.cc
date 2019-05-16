#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>

#include "strutils.hh"
#include "encgendis.hh"
#include "ppm.hh"
#include "urb.hh"
#include "zone.hh"
#include "cudamem.hh"
#include "numutils.hh"
#include "imgutils.hh"
#include "cholo.hh"

using namespace makemore;
using namespace std;

int main() {
  seedrand();
  double pi = 0.001;
  double nu = 0.001;

  unsigned int mbn = 1;
  Encgendis egd("big.proj", mbn);

  std::vector<std::string> srcimages;

  std::string srcdir = "/home/dan/aligned";
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

  double *tmpd = new double[1<<20];
  double *tmpd2 = new double[1<<20];
  uint8_t *tmpb = new uint8_t[1<<20];

  unsigned int n = 100000;
  Cholo cholo(egd.ctrlay->n);

((Megatron *)((Multitron *)egd.enc)->mt1)->activated = false;

fprintf(stderr, "starting\n");

unsigned int w = 256, h = 256;
  unsigned int i = 0;
  unsigned int which = 0;
  assert(srcimages.size() % mbn == 0);

  while (which < srcimages.size()) {
      memset(egd.ctxbuf, 0, sizeof(double) * egd.ctxlay->n);
      std::string srcfn = srcimages[which];

      std::string png = slurp(srcfn);
std::vector<std::string> tags;
      pngrgb(png, w, h, egd.tgtbuf, &tags);
      ++which;

bool isw = 0, isf = 0, ism = 0, isb = 0;
      for (auto t : tags) {
if (t == "male") { ism = 1; }
if (t == "black") { isb = 1; }
if (t == "female") { isf = 1; }
if (t == "white") { isw = 1; }
}

//if (!isf) continue;

    egd.encode();

    cholo.observe(egd.ctrbuf);


fprintf(stderr, "i=%d\n", i);

    ++i;
  }


cholo.finalize();
  cholo.save(stdout);

fprintf(stderr, "done\n");
}

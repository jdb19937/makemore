#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>

#include <string>
#include <vector>
#include <algorithm>

#include "numutils.hh"
#include "imgutils.hh"
#include "strutils.hh"
#include "partrait.hh"

using namespace std;
using namespace makemore;

double pld(double x0, double y0, double x1, double y1, double x2, double y2) {
  double d = 0;
  d = -((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1);
  d /= sqrt((y2 - y1) * (y2 - y1) + (x2 - x1) * (x2 - x1));
  return d;
}

int main(int argc, char **argv) {
  std::string srcdir = argv[1];
  std::string dstdir = argv[2];

  std::vector<std::string> srcimages;

  struct dirent *de;
  DIR *dp = opendir(srcdir.c_str());
  assert(dp);
  while ((de = readdir(dp))) {
    if (*de->d_name == '.')
      continue;
    srcimages.push_back(de->d_name);
  }
  closedir(dp);
  std::sort(srcimages.begin(), srcimages.end());
  assert(srcimages.size());

  for (auto srcfn : srcimages) {
    Partrait par;
    par.load(srcdir + "/" + srcfn);
    if (!par.has_pose())
      continue;

    Partrait newpar(256, 256);
    newpar.set_pose(Pose(128, 160, 64, 0.0, 1.0, 0.0));

    par.warp(&newpar);

    newpar.save(dstdir + "/" + srcfn);

    fprintf(stderr, "aleyened %s/%s -> %s/%s\n", srcdir.c_str(), srcfn.c_str(), dstdir.c_str(), srcfn.c_str());
  }

  return 0;
}

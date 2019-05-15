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
#include "autoposer.hh"
#include "cholo.hh"
#include "warp.hh"
#include "triangle.hh"
#include "partrait.hh"

using namespace makemore;
using namespace std;

int main() {
  seedrand();

  unsigned int mbn = 1;
  Autoposer autoposer("autoposer.proj");

  std::vector<std::string> srcimages;

  std::vector<std::string> srcdirs;
  srcdirs.push_back("/home/dan/makemore/cam");
  srcdirs.push_back("/home/dan/makemore/allmugs");
  for (auto srcdir : srcdirs) {
    struct dirent *de;
    DIR *dp = opendir(srcdir.c_str());
    assert(dp);
    while ((de = readdir(dp))) {
      if (*de->d_name == '.')
        continue;
      srcimages.push_back(srcdir + "/" + de->d_name);
    }
    closedir(dp);
  }
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
    Pose srcpose = srcpar.get_pose();

    Pose pose = Pose::STANDARD;
    pose.center.x += randrange(-40.0, 40.0);
    pose.center.y += randrange(-40.0, 40.0);
    pose.scale = 64.0 + randrange(-20.0, 20.0);
    pose.stretch = srcpose.stretch * randrange(0.95, 1.05);
    pose.angle = srcpose.angle + randrange(-0.5, 0.5);
    pose.skew = srcpose.skew + randrange(-0.05, 0.05);

    Partrait par(256, 256);
    par.set_pose(pose);
    srcpar.warp(&par);

    autoposer.observe(par, 0.00002);

    if (i % 100 == 0) {
      autoposer.report("burnseg");
fprintf(stderr, "saving\n");
      autoposer.save();
fprintf(stderr, "saved\n");
    }
    ++i;
  }
}


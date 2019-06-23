#include "project.hh"
#include "topology.hh"
#include "random.hh"
#include "cudamem.hh"
#include "project.hh"

#include "ppm.hh"
#include "warp.hh"
#include "partrait.hh"
#include "encgen.hh"
#include "autoposer.hh"
#include "imgutils.hh"

#include <math.h>

#include <map>

using namespace makemore;

int main(int argc, char **argv) {
  seedrand();

  Autoposer autoposer("bestposer.proj");
  Autoposer autoposer2("newposer.proj");
  Encgen egd("bigsham.proj", 1);

  Partrait par;

Pose curpose;
bool first = 1;

  while (par.read_ppm(stdin)) {

if (first) {
  curpose = Pose::STANDARD;
  curpose.scale = 64;
  curpose.center.x = (double)par.w / 2.0;
  curpose.center.y = (double)par.h / 2.0;
  par.set_pose(curpose);
}

first = 0;
    Pose lastpose = par.get_pose();

    double cdrift = 0.2;
    double sdrift = 0.5;
    double drift = 0.5;
    curpose.center.x = (1.0 - cdrift) * lastpose.center.x + cdrift * (par.w / 2.0);
    curpose.center.y = (1.0 - cdrift) * lastpose.center.y + cdrift * (par.h / 2.0);
    curpose.scale = (1.0 - sdrift) * lastpose.scale + sdrift * 64.0;

    curpose.angle = (1.0 - drift) * lastpose.angle;
    curpose.stretch = (1.0 - drift) * lastpose.stretch + 1.0 * drift;
    curpose.skew = (1.0 - drift) * lastpose.skew;

    if (curpose.center.x < 0) curpose.center.x = 0;
    if (curpose.center.x >= par.w) curpose.center.x = par.w - 1;
    if (curpose.center.y < 0) curpose.center.y = 0;
    if (curpose.center.y >= par.h) curpose.center.y = par.h - 1;
    if (curpose.skew > 0.1) curpose.skew = 0.1;
    if (curpose.skew < -0.1) curpose.skew = -0.1;
    if (curpose.scale > 128.0) curpose.scale = 128.0;
    if (curpose.scale < 32.0) curpose.scale = 32.0;
    if (curpose.angle > 1.0) curpose.angle = 1.0;
    if (curpose.angle < -1.0) curpose.angle = -1.0;
    if (curpose.stretch > 1.2) curpose.stretch = 1.2;
    if (curpose.stretch < 0.8) curpose.stretch = 0.8;

    par.set_pose(curpose);

    autoposer.autopose(&par);
    autoposer2.autopose(&par);
    autoposer2.autopose(&par);

    Partrait stdpar(256, 256);
    stdpar.set_pose(Pose::STANDARD);
    par.warp(&stdpar);

    memset(egd.ctxbuf, 0, egd.ctxlay->n * sizeof(double));

    stdpar.make_sketch(egd.ctxbuf);

    Hashbag hb;
    hb.add("white");
    hb.add("male");
    memcpy(egd.ctxbuf + 192, hb.vec, sizeof(double) * 64);

    egd.ctxbuf[256] = par.get_tag("angle", 0.0);
    egd.ctxbuf[257] = par.get_tag("stretch", 1.0);
    egd.ctxbuf[258] = par.get_tag("skew", 0.0);


    assert(egd.tgtlay->n == stdpar.w * stdpar.h * 3);
    rgblab(stdpar.rgb, egd.tgtlay->n, egd.tgtbuf);
    egd.encode();
    egd.generate();
    labrgb(egd.tgtbuf, egd.tgtlay->n, stdpar.rgb);
    stdpar.warpover(&par);

    par.write_ppm(stdout);
  }
  return 0;
}

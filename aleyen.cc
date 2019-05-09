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
#include "warp.hh"

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

  unsigned int w = 400, h = 400;
  uint8_t *brgb = new uint8_t[w * h * 3];
  double *drgb = new double[w * h * 3];
  uint8_t *brgbout = new uint8_t[256 * 256 * 3];
  double *drgbout = new double[256 * 256 * 3];

  for (auto srcfn : srcimages) {
    std::string png = slurp(srcdir + "/" + srcfn);
    vector<string> tags;
    pngrgb(png, w, h, brgb, &tags);
    btodv(brgb, drgb, w * h * 3);

    bool got_left = 0, got_right = 0;
    bool got_mouth = 0;
    int x0 = -1, x1 = -1, y0 = -1, y1 = -1;
    int x2 = -1, y2 = -2;
    for (auto tag : tags) {
      const char *t = tag.c_str();
      if (!strncmp(t, "left_eye:", 9)) {
        t += 9;
        x0 = atoi(t);
        assert(t = strchr(t, ','));
        t += 1;
        y0 = atoi(t);
        got_left = true;
      } else if (!strncmp(t, "right_eye:", 10)) {
        t += 10;
        x1 = atoi(t);
        assert(t = strchr(t, ','));
        t += 1;
        y1 = atoi(t);
        got_right = true;
      } else if (!strncmp(t, "mouth:", 6)) {
        t += 6;
        x2 = atoi(t);
        assert(t = strchr(t, ','));
        t += 1;
        y2 = atoi(t);
        got_mouth = true;
      }
    }
    if (!got_left || !got_right || !got_mouth)
      continue;
    assert(x0 >= 0);
    assert(x0 < w);
    assert(x1 >= 0);
    assert(x1 < w);
    assert(x2 >= 0);
    assert(x2 < w);
    assert(y0 >= 0);
    assert(y0 < h);
    assert(y1 >= 0);
    assert(y1 < h);
    assert(y2 >= 0);
    assert(y2 < h);


    double dx = x1 - x0;
    double dy = y1 - y0;

//    double de = sqrt(dx * dx + dy * dy);
//    double dm = pld(x2, y2, x0, y0, x1, y1);
//    assert(de > 0);
//    assert(dm > 0);

//    double ex = dy * dm / de;
//    double ey = -dx * dm / de;

//    int x3 = x0 - ex;
//    int y3 = y0 - ey;

    int x3 = x2 - 0.5 * dx;
    int y3 = y2 - 0.5 * dy;
    double ex = x0 - x3;
    double ey = y0 - y3;

    x0 += -dx + ex * 1.5;
    y0 += -dy + ey * 1.5;

    x1 += dx + ex * 1.5;
    y1 += dy + ey * 1.5;

    x3 += -dx - ex * 1.0;
    y3 += -dy - ey * 1.0;

    int px0, py0, px1, py1, px2, py2;
    kwarp(drgb, w, h, x0, y0, x1, y1, x3, y3, &px0, &py0, &px1, &py1, &px2, &py2, 256, 256, drgbout);
    dtobv(drgbout, brgbout, 256 * 256 * 3);

    char buf[256];
    sprintf(buf, "frame:%d,%d,%d,%d,%d,%d", px0, py0, px1, py1, px2, py2);
 fprintf(stderr, "%s\n", buf);
    tags.push_back(buf);
  
    png = "";
    rgbpng(brgbout, 256, 256, &png, &tags);
  
    spit(png, dstdir + "/" + srcfn);
    fprintf(stderr, "aleyened %s/%s -> %s/%s\n", srcdir.c_str(), srcfn.c_str(), dstdir.c_str(), srcfn.c_str());
  }

  return 0;
}

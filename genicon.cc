#include <assert.h>
#include <math.h>

#include "ppm.hh"
#include "random.hh"

using namespace makemore;

int main(int argc, char **argv) {
  int w = 58;
  int h = 58;

  PPM ppm(w, h, 0);

  if (!strcmp(argv[1], "spectrum")) {
    for (unsigned int y = 0; y < h; ++y) {
      for (unsigned int x = 0; x < w; ++x) {
        double a = (double)x/(double)w;
        double b = (double)y/(double)h;
        double l = 0.6;
  
        labtorgb(l, a, b,
          ppm.data + y*w*3 + x*3 + 0,
          ppm.data + y*w*3 + x*3 + 1,
          ppm.data + y*w*3 + x*3 + 2
        );
      }
    }
  } else if (!strcmp(argv[1], "burn")) {
    for (unsigned int y = 0; y < h; ++y) {
      for (unsigned int x = 0; x < w; ++x) {
        int off = y*w*3 + x*3 + 0;

        if ((y/2) % 2) {
          ppm.data[off+0]=0;
          ppm.data[off+1]=0;
          ppm.data[off+2]=0;
        } else {
          ppm.data[off+0]=255;
          ppm.data[off+1]=0;
          ppm.data[off+2]=0;
        }
      }
    }
  } else if (!strcmp(argv[1], "static")) {
    for (unsigned int y = 0; y < h; ++y) {
      for (unsigned int x = 0; x < w; ++x) {
        double a = 0.5;
        double b = 0.5;
        double l = randrange(0, 1);
  
        labtorgb(l, a, b,
          ppm.data + y*w*3 + x*3 + 0,
          ppm.data + y*w*3 + x*3 + 1,
          ppm.data + y*w*3 + x*3 + 2
        );
      }
    }
  } else if (!strcmp(argv[1], "random")) {
    for (unsigned int y = 0; y < h; ++y) {
      for (unsigned int x = 0; x < w; ++x) {
        double a = randrange(0, 1);
        double b = randrange(0, 1);
        double l = randrange(0, 1);
  
        labtorgb(l, a, b,
          ppm.data + y*w*3 + x*3 + 0,
          ppm.data + y*w*3 + x*3 + 1,
          ppm.data + y*w*3 + x*3 + 2
        );
      }
    }
  } else if (!strcmp(argv[1], "fuzz")) {
    for (unsigned int y = 0; y < h; ++y) {
      for (unsigned int x = 0; x < w; ++x) {
        double a = 0.5;
        double b = 0.5;
        double rx = (double)x / (double)w;
        double ry = (double)y / (double)h;
        double d2 = (rx - 0.5) * (rx - 0.5) + (ry - 0.5) * (ry - 0.5);
        double d = sqrt(d2);
        d *= 2;
        double l;

        double col[3];
        double r = randrange(0, 3);
        if (r < 1) {
          col[0] = 255; col[1] = 0; col[2] = 0;
        } else if (r < 2) {
          col[0] = 0; col[1] = 255; col[2] = 0;
        } else {
          col[0] = 0; col[1] = 0; col[2] = 255;
        }
        ppm.data[y*w*3 + x*3 + 0] = col[0];
        ppm.data[y*w*3 + x*3 + 1] = col[1];
        ppm.data[y*w*3 + x*3 + 2] = col[2];
      }
    }
  } else if (!strcmp(argv[1], "darken")) {
    for (unsigned int y = 0; y < h; ++y) {
      for (unsigned int x = 0; x < w; ++x) {
        double a = 0.5;
        double b = 0.5;
        double rx = (double)x / (double)w;
        double ry = (double)y / (double)h;
        double d2 = (rx - 0.5) * (rx - 0.5) + (ry - 0.5) * (ry - 0.5);
        double d = sqrt(d2);

        d -= 0.2;
        d *= 2;
        if (d < 0) { d = 0; }
        if (d > 1) { d = 1; }
        double l = (1-d) * 0.0 + 0.5 * d;

        labtorgb(l, a, b,
          ppm.data + y*w*3 + x*3 + 0,
          ppm.data + y*w*3 + x*3 + 1,
          ppm.data + y*w*3 + x*3 + 2
        );
      }
    }
  } else if (!strcmp(argv[1], "brighten")) {
    for (unsigned int y = 0; y < h; ++y) {
      for (unsigned int x = 0; x < w; ++x) {
        double a = 0.5;
        double b = 0.5;
        double rx = (double)x / (double)w;
        double ry = (double)y / (double)h;
        double d2 = (rx - 0.5) * (rx - 0.5) + (ry - 0.5) * (ry - 0.5);
        double d = sqrt(d2);

        d += 0.15;
        d *= 2;
        if (d < 0) { d = 0; }
        if (d > 1) { d = 1; }
        double l = (1-d) * 1.0 + 0.5 * d;

        labtorgb(l, a, b,
          ppm.data + y*w*3 + x*3 + 0,
          ppm.data + y*w*3 + x*3 + 1,
          ppm.data + y*w*3 + x*3 + 2
        );
      }
    }
  } else if (!strcmp(argv[1], "spread")) {
    for (unsigned int y = 0; y < h; ++y) {
      for (unsigned int x = 0; x < w; ++x) {
        double a = 0.5;
        double b = 0.5;
        double rx = (double)x / (double)w;

        double l = 0.5 * (1-rx) + sigmoid(randgauss()) * rx;

        labtorgb(l, a, b,
          ppm.data + y*w*3 + x*3 + 0,
          ppm.data + y*w*3 + x*3 + 1,
          ppm.data + y*w*3 + x*3 + 2
        );
      }
    }
  } else if (!strcmp(argv[1], "fade")) {
    for (unsigned int y = 0; y < h; ++y) {
      for (unsigned int x = 0; x < w; ++x) {
        double a = 0.5;
        double b = 0.5;
        double rx = (double)x / (double)w;

        double l = sigmoid(randgauss() * 1) * (1-rx) + 0.5 * rx;

        labtorgb(l, a, b,
          ppm.data + y*w*3 + x*3 + 0,
          ppm.data + y*w*3 + x*3 + 1,
          ppm.data + y*w*3 + x*3 + 2
        );
      }
    }
  } else if (!strcmp(argv[1], "blur")) {
    for (unsigned int y = 0; y < h; ++y) {
      for (unsigned int x = 0; x < w; ++x) {
        double a = 0.5;
        double b = 0.5;
        double rx = (double)x / (double)w;

        double ry = (double)y / (double)h;
        double d2 = (rx - 0.5) * (rx - 0.5) + (ry - 0.5) * (ry - 0.5);
        double d = sqrt(d2);

        d *= 1.5;
        if (d > 1) { d = 1; }
        double l = d;

        labtorgb(l, a, b,
          ppm.data + y*w*3 + x*3 + 0,
          ppm.data + y*w*3 + x*3 + 1,
          ppm.data + y*w*3 + x*3 + 2
        );
      }
    }
  } else if (!strcmp(argv[1], "sharp")) {
    for (unsigned int y = 0; y < h; ++y) {
      for (unsigned int x = 0; x < w; ++x) {
        double a = 0.5;
        double b = 0.5;
        double rx = (double)x / (double)w;
        double ry = (double)y / (double)h;
        double d = abs(rx - 0.5) + abs(ry - 0.5);
        double l = (d > 0.5) ? 1 : 0;
  
        labtorgb(l, a, b,
          ppm.data + y*w*3 + x*3 + 0,
          ppm.data + y*w*3 + x*3 + 1,
          ppm.data + y*w*3 + x*3 + 2
        );
      }
    }
  } else {
    assert(0);
  }
  
  ppm.write(stdout);
  return 0;
}

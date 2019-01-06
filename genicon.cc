#include <assert.h>

#include "ppm.hh"
#include "random.hh"

int main(int argc, char **argv) {
  int w = 64;
  int h = 64;

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
  } else {
    assert(0);
  }
  
  ppm.write(stdout);
  return 0;
}

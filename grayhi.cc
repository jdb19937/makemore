#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include <vector>

typedef std::vector<double> Vec;

static void twiddle(const Vec &z, unsigned int w, unsigned int h, Vec *lo, Vec *hi) {
  assert(w % 2 == 0 && h % 2 == 0);
  assert(z.size() == w * h);

  unsigned int nw = w/2;
  unsigned int nh = h/2;
  lo->resize(nw * nh * 1);
  hi->resize(nw * nh * 3);

  unsigned int ilo = 0, ihi = 0;

  for (unsigned int y = 0; y < h; y += 2) {
    for (unsigned int x = 0; x < w; x += 2) {
        unsigned int p = y * w + x;

        double m = (z[p] + z[p+1] + z[p+w] + z[p+w+1]) / 4.0;
        double l = (z[p] + z[p+w]) / 2.0 - m;
        double t = (z[p] + z[p+1]) / 2.0 - m;
        double s = (z[p] + z[p+w+1]) / 2.0 - m;

        (*lo)[ilo++] = m;
        (*hi)[ihi++] = 0.5 + l/2.0;
        (*hi)[ihi++] = 0.5 + t/2.0;
        (*hi)[ihi++] = 0.5 + s/2.0;
    }
  }
}


int main(int argc, char **argv) {
  assert(argc >= 3);
  int w = atoi(argv[1]);
  int h = atoi(argv[2]);

  Vec v;
  v.resize(w * h);
  int ret = fread(v.data(), sizeof(double), w * h, stdin);
  assert(ret == w * h);

  std::vector<double> lo, hi;
  twiddle(v, w, h, &lo, &hi);
  fwrite(hi.data(), sizeof(double), hi.size(), stdout);
  return 0;
}

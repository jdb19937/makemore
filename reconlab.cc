#include <stdio.h>

#include <vector>

typedef std::vector<double> Vec;

void untwiddle(const Vec &lo, const Vec &hi, unsigned int w, unsigned int h, Vec *z) {
  assert(w % 2 == 0 && h % 2 == 0);
  z->resize(3 * w * h);

  unsigned int nw = w/2;
  unsigned int nh = h/2;
  assert(lo.size() == nw * nh * 3);
  assert(hi.size() == nw * nh * 9);

  unsigned int ilo = 0, ihi = 0;

  unsigned int w3 = w * 3;
  for (unsigned int y = 0; y < h; y += 2) {
    for (unsigned int x = 0; x < w; x += 2) {
      for (unsigned int c = 0; c < 3; ++c) {
        unsigned int p = y * w3 + x * 3 + c;

        double m = lo[ilo++];
        double l = (hi[ihi++] - 0.5) * 2.0;
        double t = (hi[ihi++] - 0.5) * 2.0;
        double s = (hi[ihi++] - 0.5) * 2.0;

        (*z)[p] = m + l + t + s;
        (*z)[p+3] = m - l + t - s;
        (*z)[p+w3] = m + l - t - s;
        (*z)[p+w3+3] = m - l - t + s;
      }
    }
  }
}
int main(int argc, char **argv) {
  assert(argc == 3);
  unsigned int w = atoi(argv[1]);
  unsigned int h = atoi(argv[2]);

  assert(w % 2 == 0 && h % 2 == 0);

  std::vector<double> hi;
  hi.resize((w * h * 3 * 3)/4);
  int ret = fread(hi.data(), sizeof(double), hi.size(), stdin);
  assert(ret == hi.size());

  std::vector<double> lo;
  lo.resize((w * h * 3) / 4);
  ret = fread(lo.data(), sizeof(double), lo.size(), stdin);
  assert(ret == lo.size());

  PPM p;
  std::vector<double> v;
  untwiddle(lo, hi, w, h, &v);
  assert(v.size() == w * h * 3);

  int ret = fwrite(v.data(), 1, w * h * 3, stdout);
  assert(ret == w * h * 3);
  return 0;
}

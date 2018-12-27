#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "ppm.hh"

int main(int argc, char **argv) {
  assert(argc == 3);
  unsigned int w = atoi(argv[1]);
  unsigned int h = atoi(argv[2]);

  std::vector<double> v;
  v.resize(w * h);
  int ret = fread(v.data(), sizeof(double), w * h, stdin);
  assert(ret == w * h);

  std::vector<double> w;
  w.resize(w * h * 3);
  for (unsigned int i = 0; i < v.size(); ++i) {
    w[i * 3] = v[i];
    w[i * 3 + 1] = 0.5;
    w[i * 3 + 2] = 0.5;
  }

  int ret = fwrite(w.data(), sizeof(double), w * h * 3, stdout);
  assert(ret == w * h * 3);
}

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "ppm.hh"

int main(int argc, char **argv) {
  assert(argc == 3);
  unsigned int w = atoi(argv[1]);
  unsigned int h = atoi(argv[2]);

  std::vector<double> v;
  v.resize(w * h * 3);
  int ret = fread(v.data(), sizeof(double), w * h * 3, stdin);
  assert(ret == w * h * 3);

  std::vector<double> u;
  u.resize(w * h);
  for (unsigned int i = 0; i < w * h; ++i)
    u[i] = v[i * 3];

  ret = fwrite(u.data(), sizeof(double), w * h, stdout);
  assert(ret == w * h);
  return 0;
}

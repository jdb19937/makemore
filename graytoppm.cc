#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "ppm.hh"

using namespace makemore;

int main(int argc, char **argv) {
  assert(argc == 3);
  unsigned int w = atoi(argv[1]);
  unsigned int h = atoi(argv[2]);

  std::vector<double> v;
  v.resize(w * h);
  int ret = fread(v.data(), sizeof(double), w * h, stdin);
  assert(ret == w * h);

  PPM p;
  p.unvectorizegray(v.data(), w, h);
  p.write(stdout);
  return 0;
}

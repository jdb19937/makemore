#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "ppm.hh"

using namespace makemore;

int main(int argc, char **argv) {
  assert(argc == 3);
  unsigned int w = atoi(argv[1]);
  unsigned int h = atoi(argv[2]);

  std::vector<uint8_t> v;
  v.resize(w * h * 3);
  int ret = fread(v.data(), sizeof(uint8_t), w * h * 3, stdin);
  assert(ret == w * h * 3);

  PPM p;
  p.unvectorize(v.data(), w, h);
  p.write(stdout);
  return 0;
}

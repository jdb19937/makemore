#include <stdio.h>
#include <ctype.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include <math.h>

#include "ppm.hh"

using namespace makemore;

int main(int argc, char **argv) {
  PPM p;
  p.read(stdin);
  std::vector<double> v;
  p.vectorize(&v);
  assert(v.size() == p.w * p.h * 3);
  int ret = fwrite(v.data(), sizeof(double), v.size(), stdout);
  assert(ret == p.w * p.h * 3);
  return 0;
}

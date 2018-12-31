#define __MAKEMORE_RANDOM_CC__ 1
#include <stdio.h>
#include <random>

#include "random.hh"

static std::default_random_engine generator;

double randgauss() {
  static std::normal_distribution<double> gaussian(0, 1);
  return gaussian(generator);
}

double randrange(double a, double b) {
  static std::uniform_real_distribution<double> uniform(a, b);
  return uniform(generator);
}

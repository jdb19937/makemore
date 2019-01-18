#define __MAKEMORE_RANDOM_CC__ 1
#include <stdio.h>
#include <sys/types.h>
#include <sys/time.h>
#include <unistd.h>

#include <random>

#include "random.hh"

#include "sha256.c"

uint64_t hash64(const uint8_t *data, unsigned int len) {
  uint8_t hash[32];
  SHA256_CTX sha;
  sha256_init(&sha);
  sha256_update(&sha, data, len);
  sha256_final(&sha, hash);

  uint64_t h;
  memcpy(&h, hash, 8);
  return h;
}

static std::mt19937 generator;

void seedrand(unsigned int n) {
  fprintf(stderr, "using fixed random seed %u\n", n);
  generator.seed(n);
}

void seedrand() {
  static std::uniform_int_distribution<unsigned int> uniform(0, 1 << 31);
  generator.seed(uniform(generator) + time(NULL));
  unsigned int n = uniform(generator) + getpid();

  fprintf(stderr, "using generated random seed %u\n", n);
  generator.seed(n);
}

double randgauss() {
  static std::normal_distribution<double> gaussian(0, 1);
  double x = gaussian(generator); // for consistent random sequences
  return gaussian(generator);
}


double randrange(double a, double b) {
  static std::uniform_real_distribution<double> uniform(a, b);
  return uniform(generator);
}

unsigned int randuint() {
  static std::uniform_int_distribution<unsigned int> uniform(0, 1UL<<31);
  return uniform(generator);
}

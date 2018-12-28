#define __MAKEMORE_RANDOM_CC__ 1
#include <stdio.h>

#include "random.hh"

struct RandomSeed {
  RandomSeed() {
    fprintf(stderr, "initializing random seed\n");
    srand(rand() + time(NULL));
    srand(rand() + getpid());
  }
};

static struct RandomSeed _randomseed;

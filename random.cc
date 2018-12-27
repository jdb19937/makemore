#define __MAKEMORE_RANDOM_CC__ 1
#include "random.hh"

struct RandomSeed {
  RandomSeed() {
    srand(rand() + time(NULL));
    srand(rand() + getpid());
  }
};

static struct RandomSeed _randomseed;

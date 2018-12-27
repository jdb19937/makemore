#ifndef __MAKEMORE_DATASET_HH__
#define __MAKEMORE_DATASET_HH__ 1

#include <sys/types.h>

struct Dataset {
  unsigned int k, n;

  off_t dataoff;
  double *dataptr;
  size_t map_size;
  void *map;
  FILE *fp;

  class Layout *lay;

  Dataset(const char *fn);
  ~Dataset();

  unsigned int pick() const;
  const double *data(unsigned int) const;
};

#endif


#ifndef __MAKEMORE_DATASET_HH__
#define __MAKEMORE_DATASET_HH__ 1

#include <sys/types.h>

struct Dataset {
  unsigned int k, n;

  size_t map_size;
  double *map;

  Dataset(const char *fn, unsigned int _k);
  ~Dataset();

  unsigned int pick() const;
  const double *data(unsigned int) const;
};

#endif


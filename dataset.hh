#ifndef __MAKEMORE_DATASET_HH__
#define __MAKEMORE_DATASET_HH__ 1

#include <stdio.h>
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
  void copy(unsigned int, double *) const;
  void encude(const unsigned int, double *data) const;

  void pick_minibatch(unsigned int mbn, unsigned int *mb) const;
  void copy_minibatch(const unsigned int *mb, unsigned int mbn, double *data) const;
  void encude_minibatch(const unsigned int *mb, unsigned int mbn, double *data) const;
};

#endif


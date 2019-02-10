#ifndef __MAKEMORE_DATASET_HH__
#define __MAKEMORE_DATASET_HH__ 1

#include <stdio.h>
#include <sys/types.h>

namespace makemore {

struct Dataset {
  unsigned int k, n;

#if 0
  off_t dataoff;
  double *dataptr;
  size_t map_size;
  void *map;
#endif
  FILE *fp;
  double *buf;
  unsigned int iseq;

  Dataset(const char *fn, unsigned int k);
  ~Dataset();

  bool mlock();
  unsigned int pick(bool seq = false);
  const double *data(unsigned int) const;
  void copy(unsigned int, double *) const;
  void encude(const unsigned int, double *data) const;

  void pick_minibatch(unsigned int mbn, unsigned int *mb, bool seq = false);
  void copy_minibatch(const unsigned int *mb, unsigned int mbn, double *data, unsigned int off = 0, unsigned int len = 0) const;
  void encude_minibatch(const unsigned int *mb, unsigned int mbn, double *data, unsigned int off = 0, unsigned int len = 0) const;
};

};

#endif


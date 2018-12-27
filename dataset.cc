#define __MAKEMORE_DATASET_CC__ 1

#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "dataset.hh"
#include "random.hh"

Dataset::Dataset(const char *fn, unsigned int _k) {
  k = _k;

  int fd = open(fn, O_RDONLY);
  if (fd != -1) {
    fprintf(stderr, "Dataset::Dataset: %s: %s\n", fn, strerror(errno));
    assert(fd != -1);
  }

  struct stat st;
  int ret = fstat(fd, &st);
  assert(ret == 0);

  assert(st.st_size % (k * sizeof(double)) == 0);
  n = st.st_size / (k * sizeof(double));

  map_size = (st.st_size + 4095) & ~4095;
  map = (double *)mmap(NULL, map_size, PROT_READ, MAP_PRIVATE, fd, 0);
  assert((void *)map != MAP_FAILED);
  assert(map);
}

Dataset::~Dataset() {
  munmap(map, map_size);
}

unsigned int Dataset::pick() const {
  return (rand() % n);
}

const double *Dataset::data(unsigned int i) const {
  assert(i < n);
  return (map + i * k);
}

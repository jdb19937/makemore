#define __MAKEMORE_DATASET_CC__ 1

#include <stdio.h>
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <netinet/in.h>

#include "dataset.hh"
#include "random.hh"
#include "cudamem.hh"

Dataset::Dataset(const char *fn, unsigned int _k) {
  k = _k;

  fp = fopen(fn, "r");
  if (!fp) {
    fprintf(stderr, "Dataset::Dataset: %s: %s\n", fn, strerror(errno));
    assert(fp);
  }

  assert(fseek(fp, 0, SEEK_SET) == 0);

  struct stat st;
  int ret = fstat(fileno(fp), &st);
  assert(ret == 0);
  assert(st.st_size % (k * sizeof(double)) == 0);
  n = st.st_size / (k * sizeof(double));

  map_size = (st.st_size + 4095) & ~4095;
#if 1
  map = mmap(NULL, map_size, PROT_READ, MAP_PRIVATE, fileno(fp), 0);
  assert(map != MAP_FAILED);
  assert(map);
#else
fprintf(stderr, "reading data\n");
  map = (double *)(new uint8_t[st.st_size]);
  ret = fread((uint8_t *)map, 1, st.st_size, fp);
fprintf(stderr, "read data ret=%d\n", ret);
  assert(ret == st.st_size);
#endif

  dataptr = (double *)map;
}

Dataset::~Dataset() {
#if 1
  munmap(map, map_size);
#else
  delete[] ((uint8_t *)map);
#endif
  fclose(fp);
}

bool Dataset::mlock() {
  int ret = ::mlock(map, map_size);
  if (ret != 0) {
    return false;
  } else {
    return true;
  }
}

unsigned int Dataset::pick() const {
  return (rand() % n);
}

void Dataset::pick_minibatch(unsigned int mbn, unsigned int *mb) const {
  for (unsigned int mbi = 0; mbi < mbn; ++mbi)
    mb[mbi] = rand() % n;
}

const double *Dataset::data(unsigned int i) const {
  assert(i < n);
  return (dataptr + i * k);
}

void Dataset::copy(unsigned int i, double *d) const {
  assert(i < n);
  memcpy(d, dataptr + i * k, k * sizeof(double));
}

void Dataset::encude(unsigned int i, double *d) const {
  assert(i < n);
  ::encude(dataptr + i * k, k, d);
}

void Dataset::copy_minibatch(const unsigned int *mb, unsigned int mbn, double *d, unsigned int off, unsigned int len) const {
  if (len == 0)
    len = k;
  assert(len - off >= k);

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    unsigned int i = mb[mbi];
    assert(i < n);
    memcpy(d + mbi * len + off, dataptr + i * k, k * sizeof(double));
  }
}

void Dataset::encude_minibatch(const unsigned int *mb, unsigned int mbn, double *d, unsigned int off, unsigned int len) const {
  if (len == 0)
    len = k;
  assert(len - off >= k);

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    unsigned int i = mb[mbi];
    assert(i < n);
    ::encude(dataptr + i * k, k, d + mbi * len + off);
  }
}

#if DATASET_TEST_MAIN
int main(int argc, char **argv) {
  assert(argc > 2);
  Dataset ds(argv[1], atoi(argv[2]));
  return 0;
}
#endif

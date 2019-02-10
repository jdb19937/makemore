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

namespace makemore {

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

  buf = new double[k];
  iseq = 0;
}

Dataset::~Dataset() {
#if 0
#if 1
  munmap(map, map_size);
#else
  delete[] ((uint8_t *)map);
#endif
#endif
  fclose(fp);
}


unsigned int Dataset::pick(bool seq) {
  if (seq) {
    unsigned int i = iseq % n;
    ++iseq;
    iseq %= n;

    return i;
  } else {
    return (randuint() % n);
  }
}

void Dataset::pick_minibatch(unsigned int mbn, unsigned int *mb, bool seq) {
  if (seq) {
    for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
      mb[mbi] = iseq % n;
      ++iseq;
      iseq %= n;
    }
  } else {
    for (unsigned int mbi = 0; mbi < mbn; ++mbi)
      mb[mbi] = randuint() % n;
  }
}

void Dataset::copy(unsigned int i, double *d) const {
  assert(i < n);

  int ret;
  ret = fseek(fp, (uint64_t)i * (uint64_t)k * sizeof(double), SEEK_SET);
  assert(ret == 0);
  ret = fread(d, sizeof(double), k, fp);
  assert(ret == k);
}

void Dataset::encude(unsigned int i, double *d) const {
  assert(i < n);
  int ret;
  ret = fseek(fp, (uint64_t)i * (uint64_t)k * sizeof(double), SEEK_SET);
  assert(ret == 0);
  ret = fread(buf, sizeof(double), k, fp);
  assert(ret == k);
  makemore::encude(buf, k, d);
}

void Dataset::copy_minibatch(const unsigned int *mb, unsigned int mbn, double *d, unsigned int off, unsigned int len) const {
  int ret;
  if (len == 0)
    len = k;
  assert(len - off >= k);

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    unsigned int i = mb[mbi];
    assert(i < n);

    ret = fseek(fp, (uint64_t)i * (uint64_t)k * sizeof(double), SEEK_SET);
    assert(ret == 0);
    ret = fread(d + mbi * len + off, sizeof(double), k, fp);
    assert(ret == k);
  }
}

void Dataset::encude_minibatch(const unsigned int *mb, unsigned int mbn, double *d, unsigned int off, unsigned int len) const {
  int ret;
  if (len == 0)
    len = k;
  assert(len - off >= k);

  for (unsigned int mbi = 0; mbi < mbn; ++mbi) {
    unsigned int i = mb[mbi];
    assert(i < n);

    ret = fseek(fp, (uint64_t)i * (uint64_t)k * sizeof(double), SEEK_SET);
    assert(ret == 0);
    ret = fread(buf, sizeof(double), k, fp);
    assert(ret == k);

    makemore::encude(buf, k, d + mbi * len + off);
  }
}

}

#if DATASET_TEST_MAIN
int main(int argc, char **argv) {
  assert(argc > 2);
  Dataset ds(argv[1], atoi(argv[2]));
  return 0;
}
#endif

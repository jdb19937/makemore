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
#include "layout.hh"
#include "random.hh"

Dataset::Dataset(const char *fn) {
  fp = fopen(fn, "r");
  if (!fp) {
    fprintf(stderr, "Dataset::Dataset: %s: %s\n", fn, strerror(errno));
    assert(fp);
  }

  assert(fp);
  lay = load_new<Layout>(fp);
  k = lay->n;
  uint32_t on;
  assert(1 == fread(&on, 4, 1, fp));
  n = ntohl(on);
  dataoff = ftell(fp);

  assert(dataoff % sizeof(double) == 0);
  dataoff /= sizeof(double);

  assert(fseek(fp, 0, SEEK_SET) == 0);

  struct stat st;
  int ret = fstat(fileno(fp), &st);
  assert(ret == 0);
  // fprintf(stderr, "sz=%lu off=%u k=%u n=%u\n", st.st_size, dataoff, k, n);
  assert(st.st_size == (dataoff + k * n) * sizeof(double));

  map_size = (st.st_size + 4095) & ~4095;
  map = mmap(NULL, map_size, PROT_READ, MAP_PRIVATE, fileno(fp), 0);
  assert(map != MAP_FAILED);
  assert(map);

  dataptr = (double *)map + dataoff;
}

Dataset::~Dataset() {
  munmap(map, map_size);
  fclose(fp);
}

unsigned int Dataset::pick() const {
  return (rand() % n);
}

const double *Dataset::data(unsigned int i) const {
  assert(i < n);
  return (dataptr + i * k);
}

#if DATASET_TEST_MAIN
int main(int argc, char **argv) {
  assert(argc > 1);
  Dataset ds(argv[1]);
  return 0;
}
#endif

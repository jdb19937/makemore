#include <assert.h>

#include "parson.hh"

int main(int argc, char **argv) {
  assert(argc == 3);
  const char *parsons_fn = argv[1];
  unsigned int n_parsons = (unsigned)atoi(argv[2]);

  fprintf(stderr, "creating parsons_fn=%s n_parsons=%u\n", parsons_fn, n_parsons);
  ParsonDB::create(parsons_fn, n_parsons);
  fprintf(stderr, "created parsons_fn=%s n_parsons=%u\n", parsons_fn, n_parsons);
  return 0;
}

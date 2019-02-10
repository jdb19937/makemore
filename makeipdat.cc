#include <assert.h>

#include "ipdb.hh"

using namespace makemore;

int main(int argc, char **argv) {
  assert(argc == 3);
  const char *ips_fn = argv[1];
  unsigned int n_ips = (unsigned)atoi(argv[2]);

  fprintf(stderr, "creating ips_fn=%s n_ips=%u\n", ips_fn, n_ips);
  IPDB::create(ips_fn, n_ips);
  fprintf(stderr, "created ips_fn=%s n_ips=%u\n", ips_fn, n_ips);
  return 0;
}

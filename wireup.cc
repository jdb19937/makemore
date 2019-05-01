#include <stdio.h>
#include <stdlib.h>

#include "wiring.hh"
#include "layout.hh"
#include "random.hh"

using namespace makemore;

int usage() {
  fprintf(stderr, "Usage: wireup [--minv minv] [--maxv maxv] [--reflect] in.lay out.lay\n");
  return 1;
}

int main(int argc, char **argv) {
  unsigned int minv = 0, maxv = -1;
  bool reflect = false;

  ++argv;
  --argc;
  if (argc < 1)
    return usage();

  while (*argv[0] == '-') {
    const char *arg = argv[0];

    if (!strcmp(arg, "--minv")) {
      ++argv;
      --argc;
      if (argc < 1)
        return usage();

      minv = (unsigned int)atoi(argv[0]);

    } else if (!strcmp(arg, "--maxv")) {
      ++argv;
      --argc;
      if (argc < 1)
        return usage();

      maxv = (unsigned int)atoi(argv[0]);

    } else if (!strcmp(arg, "--reflect")) {
      reflect = true;

    } else {
      return usage();
    }

    ++argv;
    --argc;
    if (argc < 1)
      return usage();
  }

  if (argc != 2)
    return usage();
  const char *ifn = argv[0];
  const char *ofn = argv[1];

  seedrand();

  Layout *il = new Layout;
  il->load_file(ifn);

  Layout *ol = new Layout;
  ol->load_file(ofn);

  Wiring *wire = new Wiring();
  wire->wireup(il, ol, minv, maxv, reflect);
  wire->save(stdout);
  return 0;
}


#include "wiring.hh"
#include "layout.hh"

int main(int argc, char **argv) {
  assert(argc >= 3);
  const char *ifn = argv[1];
  const char *ofn = argv[2];

  unsigned int minv = 0;
  if (argc >= 4)
    minv = (unsigned int)atoi(argv[3]);

  unsigned int maxv = -1;
  if (argc >= 5)
    maxv = (unsigned int)atoi(argv[4]);

  Layout *il = load_new<Layout>(ifn);
  Layout *ol = load_new<Layout>(ofn);

  Wiring *wire = new Wiring();
  wire->wireup(il, ol, minv, maxv);
  wire->save(stdout);
  return 0;
}


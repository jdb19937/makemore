#include "topology.hh"

using namespace makemore;

int main(int argc, char **argv) {
  seedrand();
  Topology top;

  assert(argc >= 2);
  for (int i = 1; i < argc; ++i) {
    Wiring w;
    w.load_file(argv[i]);
    top.addwire(w);
  }

  fprintf(stderr, "nweights=%u\n", top.nweights);
  top.save(stdout);
  return 0;
}


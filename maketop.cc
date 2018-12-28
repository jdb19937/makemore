#include "topology.hh"

int main(int argc, char **argv) {
  Topology top;

  assert(argc >= 2);
  for (int i = 1; i < argc; ++i) {
    Wiring w;
    w.load_file(argv[i]);
    top.addwire(w);
  }

  top.save(stdout);
  return 0;
}


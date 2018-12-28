#include "topology.hh"

int main(int argc, char **argv) {
  unsigned int npass = 0;
  if (argc >= 3 && !strcmp(argv[1], "-npass")) {
    npass = atoi(argv[2]);
    argc -= 2;
    argv += 2;
  }

  Topology top(npass);

  assert(argc >= 2);
  for (int i = 1; i < argc; ++i) {
    Wiring w;
    w.load_file(argv[i]);
    top.addwire(w);
  }

  fprintf(stderr, "nweights=%u npass=%u\n", top.nweights, top.npass);
  top.save(stdout);
  return 0;
}


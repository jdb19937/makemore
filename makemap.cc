#include "project.hh"
#include "topology.hh"
#include "multitron.hh"
#include "random.hh"

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "Usage: makemap file.top file.map\n");
    exit(1);
  }

  Topology *top = new Topology;
  const char *topfn = argv[1];
  top->load_file(topfn);

  const char *mapfn = argv[2];
  Multitron *tron = new Multitron(*top, 1, mapfn);
  tron->randomize();
  tron->sync(1);

  delete tron;
  delete top;
  return 0;
}

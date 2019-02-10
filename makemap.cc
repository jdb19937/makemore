#include "project.hh"
#include "topology.hh"
#include "multitron.hh"
#include "random.hh"

using namespace makemore;

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "Usage: makemap file.top file.map [dispa [dispb]]\n");
    exit(1);
  }

  Topology *top = new Topology;
  const char *topfn = argv[1];
  top->load_file(topfn);

  const char *mapfn = argv[2];
  Multitron *tron = new Multitron(*top, 1, mapfn);

  double dispa = 1.0;
  if (argc > 3)
    dispa = strtod(argv[3], NULL);
  double dispb = dispa;
  if (argc > 4)
    dispb = strtod(argv[4], NULL);

  tron->randomize(dispa, dispb);
  tron->sync(1);

  delete tron;
  delete top;
  return 0;
}

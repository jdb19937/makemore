#include "project.hh"
#include "topology.hh"
#include "network.hh"
#include "random.hh"

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "Usage: makenet file.top file.net\n");
    exit(1);
  }

  Topology *top = new Topology;
  const char *topfn = argv[1];
  top->load_file(topfn);

  const char *netfn = argv[2];
  Network *net = new Network(top, 1, netfn);
  net->randomize();

  delete net;
  delete top;
  return 0;
}

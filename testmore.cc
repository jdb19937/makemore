#include "project.hh"
#include "topology.hh"
#include "random.hh"

int main() {
#if 0
  Topology *enctop = new Topology;
  enctop->load_file("gen8.proj/enc.top");

  Network *encnet = new Network(enctop, 0, 1, "gen8.proj/enc.net");
  encnet->tron->randomize();
#endif

  Project *p = new Project("gen8.proj");
  return 0;
}

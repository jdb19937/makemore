#include "project.hh"
#include "topology.hh"
#include "network.hh"
#include "random.hh"

int main() {
  Topology *enctop = new Topology;
  enctop->load_file("gen8.proj/enc.top");

  Network *encnet = new Network(enctop, 1, "gen8.proj/enc.net");
  encnet->randomize();
  return 0;
}

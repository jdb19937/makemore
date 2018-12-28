#ifndef __MAKEMORE_TOPOLOGY_HH__
#define __MAKEMORE_TOPOLOGY_HH__ 1

#include "layout.hh"
#include "wiring.hh"

#include <vector>

struct Topology : Persist {
  std::vector<Layout *> layouts;
  std::vector<Wiring *> wirings;

  Topology();
  ~Topology();
};

#endif

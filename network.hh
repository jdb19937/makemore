#ifndef __MAKEMORE_NETWORK_HH__
#define __MAKEMORE_NETWORK_HH__ 1

#include "layout.hh"
#include "wiring.hh"
#include "topology.hh"

#include <vector>

struct Network {
  Network();
  ~Network();

  virtual void load(FILE *fp);
  virtual void save(FILE *fp) const;
};

#endif

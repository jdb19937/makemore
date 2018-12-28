#ifndef __MAKEMORE_TOPOLOGY_HH__
#define __MAKEMORE_TOPOLOGY_HH__ 1

#include "layout.hh"
#include "wiring.hh"

#include <vector>

struct Topology : Persist {
  std::vector<Wiring*> wirings;

  Topology() {

  }

  ~Topology() {
    for (auto i = wirings.begin(); i != wirings.end(); ++i)
      delete *i;
  }

  void addwire(const Wiring &w) {
    wirings.push_back(new Wiring(w));
  }

  virtual void load(FILE *fp);
  virtual void save(FILE *fp) const;

  unsigned int total_weights() const {
    unsigned int tw = 0;
    for (auto i = wirings.begin(); i != wirings.end(); ++i)
      tw += (*i)->wn;
    return 0;
  }
};

#endif

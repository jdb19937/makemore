#ifndef __MAKEMORE_TOPOLOGY_HH__
#define __MAKEMORE_TOPOLOGY_HH__ 1

#include "layout.hh"
#include "wiring.hh"

#include <vector>

namespace makemore {

struct Topology : Persist {
  unsigned int nweights;
  std::vector<Wiring*> wirings;

  Topology() {
    nweights = 0;
  }

  ~Topology() {
    for (auto i = wirings.begin(); i != wirings.end(); ++i)
      delete *i;
  }

  void addwire(const Wiring &w) {
    nweights += w.wn + w.outn;
    wirings.push_back(new Wiring(w));
  }

  virtual void load(FILE *fp);
  virtual void save(FILE *fp) const;
};

}

#endif

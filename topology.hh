#ifndef __MAKEMORE_TOPOLOGY_HH__
#define __MAKEMORE_TOPOLOGY_HH__ 1

#include "layout.hh"
#include "wiring.hh"

#include <vector>

struct Topology : Persist {
  std::vector<Layout*> layouts;
  std::vector<Wiring*> wirings;

  Topology();

  ~Topology() {
    for (auto i = layouts.begin(); i != layouts.end(); ++i)
      delete *i;
    for (auto i = wirings.begin(); i != wirings.end(); ++i)
      delete *i;
  }

  void inlayout(const Layout *inl) {
    layouts.push_back(new Layout(*inl));
  }

  void wireup(const Layout *outl, unsigned int minv, unsigned int maxv) {
    assert(layouts.size());
    Layout *inl = layouts[layouts.size() - 1];

    Wiring *wire = new Wiring();
    wire->wireup(inl, outl, minv, maxv);
    layouts.push_back(new Layout(*outl));
    wirings.push_back(wire);
  }

  virtual void load(FILE *fp);
  virtual void save(FILE *fp) const;
};

#endif

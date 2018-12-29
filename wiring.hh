#ifndef __MAKEMORE_WIRING_HH__
#define __MAKEMORE_WIRING_HH__ 1

#include "layout.hh"

#include <vector>
#include <set>

struct Wiring : Persist {
  unsigned int inn, outn, wn;
  std::set<std::pair<unsigned int, unsigned int> > connected;

  Wiring();
  ~Wiring();

  Wiring(const Wiring &wire) {
    inn = wire.inn;
    outn = wire.outn;
    wn = wire.wn;
    connected = wire.connected;
  }

  void wireup(
    const Layout *_inl,
    const Layout *_outl,
    unsigned int minv = 0,
    unsigned int maxv = 65536
  );
 
  virtual void load(FILE *fp);
  virtual void save(FILE *fp) const;
};

#endif

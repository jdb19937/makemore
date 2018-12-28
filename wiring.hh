#ifndef __MAKEMORE_WIRING_HH__
#define __MAKEMORE_WIRING_HH__ 1

#include "layout.hh"

#include <vector>

struct Wiring : Persist {
  unsigned int inn, outn, wn;
  std::vector< std::vector<unsigned int> > moi, mio, miw, mow;

  Wiring();
  ~Wiring();

  Wiring(const Wiring &wire) {
    inn = wire.inn;
    outn = wire.outn;
    wn = wire.wn;
    moi = wire.moi;
    mio = wire.mio;
    miw = wire.miw;
    mow = wire.mow;
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

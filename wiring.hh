#ifndef __MAKEMORE_WIRING_HH__
#define __MAKEMORE_WIRING_HH__ 1

#include "layout.hh"

#include <vector>

struct Wiring : Persist {
  unsigned int inn, outn;

  Wiring(
    const Layout *_inl,
    const Layout *_outl,
    unsigned int minv = 0,
    unsigned int maxv = 65536
  );

  ~Wiring();
 
  unsigned int wn;
  std::vector< std::vector<unsigned int> > moi, mio, miw, mow;

  virtual void load(FILE *fp);
  virtual void save(FILE *fp) const;
};

#endif

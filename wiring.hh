#ifndef __WIRING_HH__
#define __WIRING_HH__ 1

#include "layout.hh"

#include <vector>

struct Wiring : Persist {
  const Layout *inl, *outl;
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

  void load(FILE *fp) { assert(!"todo"); }
  void save(FILE *fp) { assert(!"todo"); }
};

#endif

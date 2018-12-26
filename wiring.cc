#define __WIRING_CC 1
#include "wiring.hh"

#include <stdio.h>
#include <math.h>

#include <vector>
#include <map>

using namespace std;

Wiring::Wiring(const Layout *_inl, const Layout *_outl, unsigned int minv, unsigned int maxv) {
  inl = _inl;
  outl = _outl;

  inn = inl->n;
  outn = outl->n;

  moi.resize(outn);
  mow.resize(outn);
  mio.resize(inn);
  miw.resize(inn);

  const double *inx = inl->x;
  const double *iny = inl->y;
  const double *inr = inl->r;
  const double *outx = outl->x;
  const double *outy = outl->y;
  const double *outr = outl->r;

  wn = 0;

  for (unsigned int outi = 0; outi < outn; ++outi) {
    multimap<double, unsigned int> dini;
    for (unsigned int ini = 0; ini < inn; ++ini) {
      double dx = outx[outi] - inx[ini];
      double dy = outy[outi] - iny[ini];
      double d = sqrt(dx * dx + dy * dy);

      if (inr)
        d -= inr[ini];
      if (inr)
        d -= outr[ini];

      dini.insert(make_pair(d, ini));
    }

     auto q = dini.begin();
     unsigned int j = 0;
     while (q != dini.end() && j < maxv && (j < minv || q->first < 0)) {
       unsigned int ini = q->second;
       moi[outi].push_back(ini + 1);
       mio[ini].push_back(outi + 1);

       mow[outi].push_back(wn);
       miw[ini].push_back(wn);

       ++q;
       ++j;

       ++wn;
     }

     moi[outi].push_back(0);
     mow[outi].push_back(wn);
     ++wn;
  }

  for (auto q = mio.begin(); q != mio.end(); ++q) {
    q->push_back(0);
  }
}

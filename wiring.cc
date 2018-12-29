#define __MAKEMORE_WIRING_CC__ 1
#include "wiring.hh"

#include <stdio.h>
#include <stdint.h>
#include <netinet/in.h>

#include <math.h>

#include <vector>
#include <map>

using namespace std;

Wiring::Wiring() {
  inn = outn = 0;
}

Wiring::~Wiring() {

}

void Wiring::wireup(const Layout *inl, const Layout *outl, unsigned int minv, unsigned int maxv) {
  inn = inl->n;
  outn = outl->n;

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
        d -= outr[outi];

      dini.insert(make_pair(d, ini));
    }

     auto q = dini.begin();
     unsigned int j = 0;
     while (q != dini.end() && j < maxv && (j < minv || q->first < 0)) {
       unsigned int ini = q->second;
       connected.insert(make_pair(ini, outi));
       ++wn;
       ++q;
     }
  }
}

void Wiring::load(FILE *fp) {
  uint32_t tmp;
  int ret = fread(&tmp, 4, 1, fp);
  assert(ret == 1);
  inn = ntohl(tmp);

  ret = fread(&tmp, 4, 1, fp);
  assert(ret == 1);
  outn = ntohl(tmp);

  ret = fread(&tmp, 4, 1, fp);
  assert(ret == 1);
  wn = ntohl(tmp);

  for (unsigned int wi = 0; wi < wn; ++wi) {
    unsigned int ini;
    ret = fread(&ini, 4, 1, fp);
    assert(ret == 1);
    ini = ntohl(ini);

    unsigned int outi;
    ret = fread(&outi, 4, 1, fp);
    assert(ret == 1);
    outi = ntohl(outi);

    connected.insert(make_pair(ini, outi));
  }
}

void Wiring::save(FILE *fp) const {
  int ret;

  uint32_t tmp = htonl(inn);
  assert(1 == fwrite(&tmp, 4, 1, fp));

  tmp = htonl(outn);
  assert(1 == fwrite(&tmp, 4, 1, fp));

  tmp = htonl(wn);
  assert(1 == fwrite(&tmp, 4, 1, fp));

  for (auto wi = connected.begin(); wi != connected.end(); ++wi) {
    tmp = htonl(wi->first);
    ret = fwrite(&tmp, 4, 1, fp);
    assert(ret == 1);

    tmp = htonl(wi->second);
    ret = fwrite(&tmp, 4, 1, fp);
    assert(ret == 1);
  }
}


  

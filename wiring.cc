#define __MAKEMORE_WIRING_CC__ 1
#include "wiring.hh"

#include <stdio.h>
#include <stdint.h>
#include <netinet/in.h>

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
        d -= outr[outi];

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
  for (auto q = miw.begin(); q != miw.end(); ++q) {
    q->push_back(-1);
  }
}

static void _getvecvec(vector< vector<unsigned int> > &vv, FILE *fp) {
  unsigned int vvs;
  assert(1 == fread(&vvs, 4, 1, fp));
  vvs = ntohl(vvs);
  vv.resize(vvs);

  for (unsigned int vvi = 0; vvi < vv.size(); ++vvi) {
    vector<unsigned int> &v = vv[vvi];

    unsigned int vs;
    assert(1 == fread(&vs, 4, 1, fp));
    vs = ntohl(vs);
    v.resize(vs);

    vector<unsigned int> nv;
    nv.resize(v.size());
    assert(nv.size() == fread(nv.data(), sizeof(unsigned int), nv.size(), fp));

    for (unsigned int i = 0; i < nv.size(); ++i)
      v[i] = ntohl(nv[i]);
  }
}

void Wiring::load(FILE *fp) {
  inl = load_new<Layout>(fp);
  outl = load_new<Layout>(fp);
  _getvecvec(mio, fp);
  _getvecvec(miw, fp);
  _getvecvec(moi, fp);
  _getvecvec(mow, fp);
}

static void _putvecvec(const vector< vector<unsigned int> > &vv, FILE *fp) {
  unsigned int vvs = htonl(vv.size());
  assert(1 == fwrite(&vvs, 4, 1, fp));

  for (unsigned int vvi = 0; vvi < vv.size(); ++vvi) {
    const vector<unsigned int> &v = vv[vvi];
    unsigned int vs = htonl(v.size());
    assert(1 == fwrite(&vs, 4, 1, fp));

    vector<unsigned int> nv;
    nv.resize(v.size());
    for (unsigned int i = 0; i < nv.size(); ++i)
      nv[i] = htonl(v[i]);
    assert(nv.size() == fwrite(nv.data(), sizeof(unsigned int), nv.size(), fp));
  }
}

void Wiring::save(FILE *fp) const {
  inl->save(fp);
  outl->save(fp);
  _putvecvec(mio, fp);
  _putvecvec(miw, fp);
  _putvecvec(moi, fp);
  _putvecvec(mow, fp);
}


  

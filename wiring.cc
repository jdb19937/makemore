#define __MAKEMORE_WIRING_CC__ 1
#include "wiring.hh"

#include <stdio.h>
#include <stdint.h>
#include <netinet/in.h>

#include <math.h>

#include <vector>
#include <map>
#include <algorithm>

namespace makemore {
using namespace std;

Wiring::Wiring() {
  inn = outn = 0;
  wn = 0;
}

Wiring::~Wiring() {

}

void Wiring::wireup(const Layout *inl, const Layout *outl, unsigned int minv, unsigned int maxv, bool reflect) {
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
//fprintf(stderr, "wireup %u/%u\n", outi, outn);
    multimap<double, unsigned int> dini;
    for (unsigned int ini = 0; ini < inn; ++ini) {
      double dx = outx[outi] - inx[ini];
      double dy = outy[outi] - iny[ini];
      double d = sqrt(dx * dx + dy * dy);

      if (inr)
        d -= inr[ini];
      if (inr)
        d -= outr[outi];

if (d < 0)
      dini.insert(make_pair(d, ini));


      if (reflect) {
        dx = outx[outi] - (0.5 - inx[ini]);
        d = sqrt(dx * dx + dy * dy);

        if (inr)
          d -= inr[ini];
        if (inr)
          d -= outr[outi];
if (d < 0)
        dini.insert(make_pair(d, ini));
      }
    }

     auto q = dini.begin();
     unsigned int j = 0;
     while (q != dini.end() && j < maxv && (j < minv || q->first < 0)) {
       unsigned int ini = q->second;
       //connected.insert(make_pair(ini, outi));
       connected.push_back(make_pair(ini, outi));
       ++wn;
       ++q;
       ++j;
     }
  }

  sort(connected.begin(),connected.end());
  unique(connected.begin(), connected.end());

  std::vector<unsigned int> fanout, fanin;
  fanout.resize(inn);
  fanin.resize(outn);

  for (auto ci = connected.begin(); ci != connected.end(); ++ci) {
    unsigned int ini = ci->first;
    unsigned int outi = ci->second;
    fanin[outi]++;
    fanout[ini]++;
  }

  double avgfanin = 0, minfanin = -1, maxfanin = -1;
  for (unsigned int outi = 0; outi < outn; ++outi) {
    avgfanin += (double)fanin[outi] / (double)outn;
    if (fanin[outi] > maxfanin)
      maxfanin = fanin[outi];
    if (minfanin < 0 || fanin[outi] < minfanin)
      minfanin = fanin[outi];
  }
  fprintf(stderr, "avgfanin=%lf minfanin=%lf maxfanin=%lf\n", avgfanin,  minfanin, maxfanin);

  double avgfanout = 0, minfanout = -1, maxfanout = -1;
  for (unsigned int ini = 0; ini < inn; ++ini) {
    avgfanout += (double)fanout[ini] / (double)inn;
    if (fanout[ini] > maxfanout)
      maxfanout = fanout[ini];
    if (minfanout < 0 || fanout[ini] < minfanout)
      minfanout = fanout[ini];
  }
  fprintf(stderr, "avgfanout=%lf minfanout=%lf maxfanout=%lf\n", avgfanout,  minfanout, maxfanout);
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

//std::set<std::pair<unsigned int, unsigned int> > con1;
//std::vector<std::pair<unsigned int, unsigned int> > con2;

  for (unsigned int wi = 0; wi < wn; ++wi) {
    unsigned int ini;
    ret = fread(&ini, 4, 1, fp);
    assert(ret == 1);
    ini = ntohl(ini);

    unsigned int outi;
    ret = fread(&outi, 4, 1, fp);
    assert(ret == 1);
    outi = ntohl(outi);

    auto v = make_pair(ini, outi);
    connected.push_back(make_pair(ini, outi));
    //connected.insert(v);
  }

  sort(connected.begin(), connected.end());
  assert(connected.size() == wn);
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

void Wiring::_makemaps(
  std::vector< std::vector<unsigned int> > &mio,
  std::vector< std::vector<unsigned int> > &miw,
  std::vector< std::vector<unsigned int> > &moi,
  std::vector< std::vector<unsigned int> > &mow
) const {
  unsigned int wi = 0;

  for (auto ci = connected.begin(); ci != connected.end(); ++ci) {
    unsigned int inri = ci->first;
    unsigned int outri = ci->second;

    mio[inri].push_back(outri + 1);
    miw[inri].push_back(wi);
    mow[outri].push_back(wi);
    moi[outri].push_back(inri + 1);
    ++wi;
  }

  assert(wi == wn);
}

  
void Wiring::load_file(const char *fn) {
  FILE *fp = fopen(fn, "r");
  if (!fp) {
    fprintf(stderr, "%s: %s\n", fn, strerror(errno));
    assert(fp);
  }
  load(fp);
  fclose(fp);
}

}

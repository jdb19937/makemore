#define __MAKEMORE_MULTITRON_CC__ 1

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <errno.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <fcntl.h>

#include <vector>

#include "topology.hh"
#include "multitron.hh"
#include "mapfile.hh"
#include "normatron.hh"

namespace makemore {

Multitron::Multitron(const Topology &top, Mapfile *_mapfile, unsigned int _mbn, bool activated, bool normalized) : Tron(0, 0) {
  mapfile = _mapfile;
  assert(mapfile);

  mbn = _mbn;

  const std::vector<Wiring*>& wirings = top.wirings;
  assert(wirings.begin() != wirings.end());

  trons.clear();
  Megatron *prev = NULL;
  for (auto wi = wirings.begin(); wi != wirings.end(); ++wi) {
    Megatron *mt = new Megatron(*wi, mapfile, mbn);
    trons.push_back(mt);

    if (prev)
      assert(mt->inn == prev->outn);
    prev = mt;
  }

  prev->activated = activated;
  if (normalized) {
    Normatron *t = new Normatron(mapfile, prev->outrn, mbn);
    assert(t->inn == prev->outn);
    trons.push_back(t);
  }

  mt0 = trons[0];
  mt1 = trons[trons.size() - 1];

  inrn = (*wirings.begin())->inn;
  outrn = (*wirings.rbegin())->outn;

  inn = inrn * mbn;
  outn = outrn * mbn;

  assert(mt0->inn == inn);
  assert(mt1->outn == outn);
}

void Multitron::randomize(double dispersion) {
  assert(trons.size() > 0);

  for (auto ti = trons.begin(); ti != trons.end(); ++ti) {
    (*ti)->randomize(dispersion);
  }
}

Multitron::~Multitron() {
  for (auto ti = trons.begin(); ti != trons.end(); ++ti)
    delete *ti;
}


const double *Multitron::feed(const double *in, double *fin) {
  auto mi = trons.begin();
  assert(mi != trons.end());
  const double *out = (*mi)->feed(in, fin);
  double *fout = (*mi)->foutput();

  ++mi;
  while (mi != trons.end()) {
    out = (*mi)->feed(out, fout);
    fout = (*mi)->foutput();
    ++mi;
  }

  return out;
}

void Multitron::train(double nu) {
  for (int i = trons.size() - 1; i >= 0; --i) {
    trons[i]->train(nu);
  }
}

}

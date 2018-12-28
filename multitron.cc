#define __MAKEMORE_MULTITRON_CC__ 1

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <errno.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <fcntl.h>

#include "topology.hh"
#include "multitron.hh"
#include "cudamem.hh"

#include <vector>

Multitron::Multitron(const Topology &top, unsigned int _npass, unsigned int _mbn, const char *mapfn) : Tron(0, 0) {
  unsigned int twn = top.nweights;
  map_size = ((twn * sizeof(double)) + 4095) & ~4095;
  fd = -1;

  if (mapfn) {
    fd = ::open(mapfn, O_RDWR | O_CREAT, 0777);
    if (fd < 0) {
      fprintf(stderr, "%s: %s\n", mapfn, strerror(errno));
      assert(!(fd < 0));
    }

    int ret = ::ftruncate(fd, twn * sizeof(double));
    assert(ret == 0);

    map = (double *)::mmap(NULL, map_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  } else {
    map = (double *)::mmap(NULL, map_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  }

  assert(map != MAP_FAILED);
  assert(map);

  const std::vector<Wiring*>& wirings = top.wirings;
  assert(wirings.begin() != wirings.end());

  npass = _npass;
  mbn = _mbn;

  megatrons.clear();
  double *wb = map;
  for (auto wi = wirings.begin(); wi != wirings.end(); ++wi) {
    Megatron *mt = new Megatron(*wi, wb, mbn);
    megatrons.push_back(mt);
    wb += (*wi)->wn;
  }
  inrn = (*wirings.begin())->inn;
  outrn = (*wirings.rbegin())->outn + npass;

  inn = inrn * mbn;
  outn = outrn * mbn;

  passbuf = NULL;
  fpassbuf = NULL;
  if (npass > 0) {
    cumake(&passbuf, outrn * mbn);
    cumake(&fpassbuf, outrn * mbn);
  }
}

void Multitron::randomize(double disp) {
  for (auto ti = megatrons.begin(); ti != megatrons.end(); ++ti)
    (*ti)->randomize(disp);
}

Multitron::~Multitron() {
  if (passbuf)
    cufree(passbuf);
  if (fpassbuf)
    cufree(fpassbuf);

  ::munmap(map, map_size);
  if (fd >= 0)
    ::close(fd);
}


const double *Multitron::feed(const double *_in, double *_fin) {
  in = _in;
  fin = _fin;

  auto mi = megatrons.begin();
  assert(mi != megatrons.end());
  out = (*mi)->feed(in, fin);
  fout = (*mi)->foutput();

  auto pmi = mi++;
  while (mi != megatrons.end()) {
    out = (*mi)->feed(out, fout);
    fout = (*mi)->foutput();
    ++mi;
  }

  if (npass == 0)
    return out;

  cuzero(fpassbuf, mbn * outrn); 
  cucutpaste(in, out, mbn, inrn, outrn - npass, outrn, passbuf);

  return passbuf;
}

void Multitron::train(double nu) {
  if (npass > 0) {
    assert(fpassbuf);
    cucutadd(fpassbuf, mbn, outrn, outrn - npass, fout);
  }

  for (auto mi = megatrons.rbegin(); mi != megatrons.rend(); ++mi) {
    (*mi)->train(nu);
  }
}

void Multitron::sync(double t) {
  for (auto mi = megatrons.rbegin(); mi != megatrons.rend(); ++mi) {
    (*mi)->sync(t);
  }
}

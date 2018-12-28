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

Multitron::Multitron(const Topology &top, unsigned int _mbn, const char *mapfn) : Tron(0, 0) {
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

  mbn = _mbn;

  megatrons.clear();
  double *wb = map;
  for (auto wi = wirings.begin(); wi != wirings.end(); ++wi) {
    Megatron *mt = new Megatron(*wi, wb, mbn);
    megatrons.push_back(mt);
    wb += (*wi)->wn;
  }
  assert(map + twn == wb);

  inrn = (*wirings.begin())->inn;
  outrn = (*wirings.rbegin())->outn;

  inn = inrn * mbn;
  outn = outrn * mbn;
}

void Multitron::randomize(double disp) {
  for (auto ti = megatrons.begin(); ti != megatrons.end(); ++ti)
    (*ti)->randomize(disp);
}

Multitron::~Multitron() {
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

  ++mi;
  while (mi != megatrons.end()) {
    out = (*mi)->feed(out, fout);
    fout = (*mi)->foutput();
    ++mi;
  }

  return out;
}

void Multitron::train(double nu) {
  for (auto mi = megatrons.rbegin(); mi != megatrons.rend(); ++mi) {
    (*mi)->train(nu);
  }
}

void Multitron::sync(double t) {
  for (auto mi = megatrons.rbegin(); mi != megatrons.rend(); ++mi) {
    (*mi)->sync(t);
  }
}

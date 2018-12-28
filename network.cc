#define __MAKEMORE_NETWORK_CC__ 1
#include "network.hh"

#include <stdio.h>
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <netinet/in.h>

#include "dataset.hh"
#include "random.hh"
#include "cudamem.hh"
#include "topology.hh"

Network::Network(const Topology *_top, unsigned int _mbn = 1, const char *_fn = NULL) {
  top = _top;
  fn = _fn;
  mbn = _mbn;

  twn = top->total_weights();

  map_size = ((twn * sizeof(double)) + 4095) & ~4095;
  fd = -1;
  if (fn) {
    fd = ::open(fn, O_RDWR | O_CREAT, 0777);
    if (fd < 0) {
      fprintf(stderr, "%s: %s\n", fn, strerror(errno));
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

  megatrons.clear();

  double *wb = map;
  for (auto wi = top->wirings.begin(); wi != top->wirings.end(); ++wi) {
    Megatron *mt = new Megatron(*wi, wb, mbn);
    megatrons.push_back(mt);
    wb += (*wi)->wn;
  }

  tron = NULL;
  for (auto ti = megatrons.begin(); ti != megatrons.end(); ++ti) {
    if (tron) {
      tron = compositron(tron, *ti);
      compositrons.push_back(tron);
    } else {
      tron = *ti;
    }
  }

  enctron = encudatron(tron->inn);
  dectron = decudatron(tron->outn);

  cutron = tron;

  cutron = compositron(enctron, cutron);
  compositrons.push_back(cutron);

  cutron = compositron(cutron, dectron);
  compositrons.push_back(dectron);
}

void Network::randomize(double disp) {
  for (auto ti = megatrons.begin(); ti != megatrons.end(); ++ti)
    (*ti)->randomize(disp);
}

Network::~Network() {
  for (auto ti = megatrons.begin(); ti != megatrons.end(); ++ti)
    delete *ti;
  for (auto ti = compositrons.begin(); ti != compositrons.end(); ++ti)
    delete *ti;
  delete enctron;
  delete dectron;

  ::munmap(map, map_size);
  if (fd >= 0)
    ::close(fd);
}


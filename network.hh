#ifndef __MAKEMORE_NETWORK_HH__
#define __MAKEMORE_NETWORK_HH__ 1

#include "layout.hh"
#include "wiring.hh"
#include "topology.hh"
#include "megatron.hh"
#include "multitron.hh"

#include <string>
#include <vector>

struct Network {
  unsigned int inrn, outrn;

  const Topology *top;
  unsigned int mbn;

  int fd;
  const char *fn;

  size_t map_size;
  double *map;

  Network(const Topology *_top, unsigned int _mbn, const char *_fn);
  virtual ~Network();

  Multitron *tron;
};

#endif

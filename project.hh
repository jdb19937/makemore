#ifndef __MAKEMORE_PROJECT_HH__
#define __MAKEMORE_PROJECT_HH__ 1

#include "dataset.hh"
#include "layout.hh"
#include "topology.hh"
#include "multitron.hh"

#include <string>

struct Project {
  std::string dir;
  unsigned int mbn;

  Project(const char *_dir, unsigned int _mbn = 4);
  ~Project();

  Layout *sampleslay, *contextlay, *controlslay;
  Dataset *samples, *context;

  Topology *enctop, *gentop, *distop;
  Multitron *enctron, *gentron, *distron;
};

#endif


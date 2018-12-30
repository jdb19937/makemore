#ifndef __MAKEMORE_PROJECT_HH__
#define __MAKEMORE_PROJECT_HH__ 1

#include "dataset.hh"
#include "layout.hh"
#include "topology.hh"
#include "multitron.hh"

#include <string>

struct Project {
  Project(const char *_dir, unsigned int _mbn = 4);
  ~Project();

  std::string dir;
  unsigned int mbn;

  Layout *sampleslay, *contextlay, *controlslay;

  Topology *enctop, *gentop, *distop;
  Multitron *enctron, *gentron, *distron;

  Tron *encpasstron, *encgentron, *encdistron;

  // fidelity, center, confuse, discern
};

struct SimpleProject : Project {
  SimpleProject(const char *_dir, unsigned int _mbn = 4);
  ~SimpleProject();

  Dataset *samples, *context;
};

struct ZoomProject : Project {
  ZoomProject(const char *_dir, unsigned int _mbn = 4);
  ~ZoomProject();

  Layout *lofreqlay, *attrslay;
  Dataset *lofreq, *hifreq, *attrs;
};


#endif


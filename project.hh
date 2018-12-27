#ifndef __MAKEMORE_PROJECT_HH__
#define __MAKEMORE_PROJECT_HH__ 1

#include "dataset.hh"
#include "layout.hh"

#include <string>

struct Project {
  std::string dir;

  Dataset *samples;
  Dataset *context;
  Layout *control;

  Project(const char *_dir);
  ~Project();
};

#endif


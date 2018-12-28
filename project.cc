#define __MAKEMORE_PROJECT_CC__ 1

#include <string.h>
#include <assert.h>

#include "project.hh"

Project::Project(const char *_dir) {
  assert(strlen(_dir) < 4000);
  dir = _dir;
}

Project::~Project() {
}


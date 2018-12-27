#define __MAKEMORE_PROJECT_CC__ 1

#include <string.h>
#include <assert.h>

#include "project.hh"

Project::Project(const char *_dir) {
  assert(strlen(_dir) < 4000);
  dir = _dir;

  char samples_fn[4096];
  sprintf(samples_fn, "%s/samples.dat", dir.c_str());
  samples = new Dataset(samples_fn);

  char context_fn[4096];
  sprintf(context_fn, "%s/context.dat", dir.c_str());
  context = new Dataset(context_fn);
}

Project::~Project() {
  delete samples;
  delete context;
}

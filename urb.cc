#define __MAKEMORE_URB_CC__ 1
#include <assert.h>
#include <stdio.h>

#include "urb.hh"
#include "pipeline.hh"

namespace makemore {

using namespace std;

Urb::Urb(const char *_dir, unsigned int _mbn) {
  mbn = _mbn;

  assert(strlen(_dir) < 4000);
  dir = _dir;

  zone = new Zone(dir + "/main.zone");
  pipe1 = new Pipeline((dir + "/partrait.proj").c_str(), 1);
  pipex = new Pipeline((dir + "/partrait.proj").c_str(), mbn);
}

Urb::~Urb() {
  delete pipex;
  delete pipe1;
  delete zone;
}

}

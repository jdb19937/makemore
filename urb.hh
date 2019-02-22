#ifndef __MAKEMORE_URB_HH__
#define __MAKEMORE_URB_HH__ 1

#include "pipeline.hh"
#include "parson.hh"
#include "zone.hh"

namespace makemore {

struct Urb {
  unsigned int mbn;
  std::string dir;

  Zone *zone;
  Pipeline *pipe1, *pipex;

  Urb(const char *_dir, unsigned int _mbn = 8);
  ~Urb();
};

}

#endif

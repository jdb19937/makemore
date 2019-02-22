#ifndef __MAKEMORE_URB_HH__
#define __MAKEMORE_URB_HH__ 1

#include "pipeline.hh"
#include "parson.hh"
#include "zone.hh"
#include "bus.hh"

namespace makemore {

struct Urb {
  unsigned int mbn;
  std::string dir;

  Zone *zone;
  Pipeline *pipe1, *pipex;

  Urb(const char *_dir, unsigned int _mbn = 8);
  ~Urb();

  void generate(Parson *p, long min_age = 0);
  void generate(Bus *b, long min_age = 0);
};

}

#endif

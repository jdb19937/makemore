#ifndef __MAKEMORE_URB_HH__
#define __MAKEMORE_URB_HH__ 1

#include <vector>
#include <string>

#include "pipeline.hh"
#include "parson.hh"
#include "zone.hh"
#include "brane.hh"
#include "bus.hh"

namespace makemore {

struct Urb {
  unsigned int mbn;
  std::string dir;

  std::vector<Zone*> zones;
  Bus *outgoing;

  Brane *brane1;
  Pipeline *pipe1, *pipex;

  Urb(const char *_dir, unsigned int _mbn = 8);
  ~Urb();

  void generate(Parson *p, long min_age = 0);

  unsigned int tier(const Parson *x) const;
  unsigned int tier(const Zone *zone) const;
  Zone *zone(const Parson *x) const;
  Parson *find(const std::string &nom, unsigned int *tierp = NULL) const;
  Parson *import(const Parson &x, unsigned int tier = 0);

  void _busout(const Parson &x);

  bool deport(const char *nom);
  void deport(Parson *p);
};

}

#endif

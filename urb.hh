#ifndef __MAKEMORE_URB_HH__
#define __MAKEMORE_URB_HH__ 1

#include <vector>
#include <string>

#include "pipeline.hh"
#include "parson.hh"
#include "zone.hh"
#include "brane.hh"
#include "bus.hh"

#include "cholo.hh"
#include "encgendis.hh"

namespace makemore {

struct Urb {
  unsigned int mbn;
  std::string dir;

  std::vector<Zone*> zones;
  Bus *outgoing;
  std::vector<std::string> images, srcimages;

  Encgendis *egd;
  double *knobs;
  Cholo *cholo;

  Urb(const char *_dir, unsigned int _mbn = 8);
  ~Urb();

  void generate(Parson *p, long min_age = 0);

  unsigned int tier(const Parson *x) const;
  unsigned int tier(const Zone *zone) const;
  Zone *zone(const Parson *x) const;
  Parson *find(const std::string &nom, unsigned int *tierp = NULL) const;
  Parson *make(const std::string &nom, unsigned int tier = 0, unsigned int gens = 0, Parson *child = NULL, unsigned int which = 0);
  Parson *make(const char *nom, unsigned int tier = 0) {
    return make(std::string(nom), tier);
  }
  Parson *make(const Parson &x, unsigned int tier = 0);
  Parson *make(unsigned int tier = 0);

  void _busout(const Parson &x);

  bool deport(const char *nom);
  void deport(Parson *p);
  void restock(unsigned int n, std::vector<std::string> *noms = NULL);
};

}

#endif

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
#include "random.hh"
#include "superenc.hh"
#include "supergen.hh"
#include "styler.hh"
#include "autoposer.hh"

#include <map>
#include <string>

namespace makemore {

struct Urb {
  unsigned int mbn;
  std::string dir;

  std::vector<Zone*> zones;
  Zone *sks0, *sks1;
  Bus *outgoing;
  std::vector<std::string> images, srcimages;

  Autoposer *ruffposer;
  Autoposer *fineposer;

  Superenc *enc;

  std::map<std::string, Supergen *> gens;
  Supergen *default_gen;

  std::map<std::string, Styler *> stys;
  Styler *default_sty;

  Urb(const char *_dir, unsigned int _mbn = 8);
  ~Urb();

  void add_gen(const std::string &tag, const std::string &projdir);
  void add_sty(const std::string &tag, const std::string &projdir);

  Supergen *get_gen(const std::string &tag) {
    Supergen *gen = gens[tag];
    if (!gen)
      gen = default_gen;
    return gen;
  }

  Styler *get_sty(const std::string &tag) {
    Styler *sty = stys[tag];
    if (!sty)
      sty = default_sty;
    return sty;
  }

  void generate(Parson *p, long min_age = 0);

  unsigned int tier(const Parson *x) const;
  unsigned int tier(const Zone *zone) const;
  Zone *zone(const Parson *x) const;
  Parson *find(const std::string &nom, unsigned int *tierp = NULL) const;

  Parson *make(const std::string &nom, unsigned int tier = 0, unsigned int gens = 0, Parson *child = NULL, unsigned int which = randuint() % 2);
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

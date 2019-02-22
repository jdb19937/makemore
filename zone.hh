#ifndef __MAKEMORE_ZONE_HH__
#define __MAKEMORE_ZONE_HH__ 1

#include "parson.hh"

namespace makemore {

struct Zone {
  const static unsigned int nfam = 9;
  const static unsigned int nvariants = 16;

  std::string fn;
  int fd;

  Parson *db;
  unsigned int n;

  static void create(const char *_fn, unsigned int _n);
  void fill_fam(const char *nom, Parson::Nom *);

  Zone(const char *_fn);
  Zone(const std::string &s) {
    Zone(s.c_str());
  }
  ~Zone();

  Parson *pick();
  Parson *pick(const char *tag, unsigned int max_tries);
  Parson *pick(const char *tag1, const char *tag2, unsigned int max_tries);

  bool exists(const char *nom) const {
    return (find(nom) != NULL);
  }

  Parson *find(const char *nom) const;

  Parson *import(const char *nom, Parson *evicted = NULL);
};

}

#endif

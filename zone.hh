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

  Zone(const std::string &_fn);
  ~Zone();

  Parson *pick();
  Parson *pick(unsigned int max_tries);
  Parson *pick(const char *tag, unsigned int max_tries);
  Parson *pick(const char *tag1, const char *tag2, unsigned int max_tries);

  bool exists(const std::string &nom) const {
    return (find(nom) != NULL);
  }

  Parson *find(const std::string &nom) const;

  Parson *import(const Parson &x, bool *evicted = NULL, Parson *evictee = NULL);
  Parson *left_naybor(Parson *p, unsigned int tries = 32);
  Parson *right_naybor(Parson *p, unsigned int tries = 32);

  unsigned int dom(const Parson *p) const {
    long off = p - db;
    assert (off >= 0 && off < n);
    return ((unsigned int)off);
  }

  bool has(const Parson *p) const {
    long off = p - db;
    return (off >= 0 && off < n);
  }
};

}

#endif

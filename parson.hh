#ifndef __MAKEMORE_PARSON_HH__
#define __MAKEMORE_PARSON_HH__ 1

#include <stdlib.h>
#include <stdint.h>

#include <string>
#include <map>

struct Parson {
  static bool valid_nom(const char *);
  static uint64_t hash_nom(const char *nom);
  static bool female_nom(const char *);

  const static unsigned int nfrens = 16;
  const static unsigned int ntags = 0;
  const static unsigned int dim = 64;
  const static unsigned int ncontrols = 1920;
  const static unsigned int nattrs = 40;
  typedef char Nom[32];

  uint64_t hash;
  Nom nom;

  uint32_t created;
  uint32_t revised;
  uint32_t creator;
  uint32_t revisor;

  uint16_t tags[ntags];
  uint8_t attrs[nattrs];

  uint8_t controls[ncontrols];
  uint8_t target[dim * dim * 3];

  Nom frens[nfrens];

  bool exists() {
    return (nom[0] != 0);
  }

  void initialize(const char *_nom, double mean, double dev);
};

struct ParsonDB {
  std::string fn;
  int fd;

  Parson *db;
  unsigned int n;

  static void create(const char *_fn, unsigned int _n);

  ParsonDB(const char *_fn);
  ~ParsonDB();

  Parson *find(const char *pname);

  bool exists(const char *qname) {
    Parson *p = find(qname);
    return p->exists();
  }
};

#endif

#ifndef __MAKEMORE_PARSON_HH__
#define __MAKEMORE_PARSON_HH__ 1

#include <stdlib.h>
#include <stdint.h>

#include <string>
#include <map>

struct Parson {
  static bool valid_nom(const char *);
  static uint64_t hash_nom(const char *nom);
  static uint64_t hash_tag(const char *tag);
  static bool female_nom(const char *);
  static std::string bread(const char *nom0, const char *nom1, uint8_t);
  static void paren_noms(const char *, char *, char *);

  const static unsigned int nfrens = 16;
  const static unsigned int ntags = 0;
  const static unsigned int dim = 64;
  const static unsigned int ncontrols = 1920;
  const static unsigned int nattrs = 72;
  typedef char Nom[32];
  typedef char Tag[32];

  uint64_t hash;
  Nom nom;

  uint32_t created;
  uint32_t revised;
  uint32_t creator;
  uint32_t revisor;

  uint8_t target_lock;
  uint8_t control_lock;
  uint8_t _pad[6];

  Tag tags[ntags];
  uint8_t attrs[nattrs];

  double controls[ncontrols];
  double target[dim * dim * 3];

  Nom frens[nfrens];
  Nom parens[2];

  bool exists() {
    return (nom[0] != 0);
  }

  void initialize(const char *_nom, double mean, double dev);

  void add_fren(const char *fnom);
  void set_parens(const char *anom, const char *bnom);

  bool fraternal(const Parson *p) {
    if (*parens[0] && *p->parens[0] && !strcmp(parens[0], p->parens[0]))
      return true;
    if (*parens[1] && *p->parens[0] && !strcmp(parens[1], p->parens[0]))
      return true;
    if (*parens[0] && *p->parens[1] && !strcmp(parens[0], p->parens[1]))
      return true;
    if (*parens[1] && *p->parens[1] && !strcmp(parens[1], p->parens[1]))
      return true;
    return false;
  }
};

struct ParsonDB {
  const static unsigned int nfam = 9;

  std::string fn;
  int fd;

  Parson *db;
  unsigned int n;

  static void create(const char *_fn, unsigned int _n);
  void fill_fam(const char *nom, Parson::Nom *);

  ParsonDB(const char *_fn);
  ~ParsonDB();

  Parson *find(const char *pname);

  bool exists(const char *qname) {
    Parson *p = find(qname);
    return p->exists();
  }
};

#endif

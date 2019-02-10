#ifndef __MAKEMORE_URB_HH__
#define __MAKEMORE_URB_HH__ 1

#include "parson.hh"

namespace makemore {

struct Urb {
  const static unsigned int nfam = 9;
  const static unsigned int nvariants = 16;

  std::string fn;
  int fd;

  Parson *db;
  unsigned int n;

  static void create(const char *_fn, unsigned int _n);
  void fill_fam(const char *nom, Parson::Nom *);

  Urb(const char *_fn);
  ~Urb();

  Parson *find(const char *pname);
  Parson *pick();
  Parson *pick(bool male);
  Parson *pick(bool male, bool old);

  bool exists(const char *qname) {
    Parson *p = find(qname);
    return p->exists();
  }
};

}

#endif

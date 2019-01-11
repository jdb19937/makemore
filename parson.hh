#ifndef __MAKEMORE_PARSON_HH__
#define __MAKEMORE_PARSON_HH__ 1

#include <stdlib.h>
#include <stdint.h>

#include <string>
#include <map>

struct Parson {
  const static unsigned int nfrens = 16;
  const static unsigned int ntags = 2;
  const static unsigned int dim = 64;
  const static unsigned int nctrls = 1920;
  typedef char Name[32];

  uint64_t hash;
  Name name;

  uint32_t created;
  uint32_t revised;
  uint32_t creator;
  uint32_t revisor;

  uint16_t tags[ntags];
  uint64_t attrs;

  uint8_t controls[nctrls];
  uint8_t target[dim * dim * 3];

  Name frens[nfrens];

  bool exists() {
    return (name[0] != 0);
  }
};

struct ParsonDB {
  std::string fn;
  int fd;

  ParsonDB(const char *_fn);
  ~ParsonDB();

  Parson *find(const char *pname);

  bool exists(const char *qname) {
    Parson *p = find(qname);
    return p->exists();
  }
};

#endif

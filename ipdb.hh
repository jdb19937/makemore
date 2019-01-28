#ifndef __MAKEMORE_IPDB__
#define __MAKEMORE_IPDB__ 1

#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include <string>
#include <map>

struct IPDB {
  struct Entry {
    uint32_t ip;
    time_t updated;
    double burndebt;
  };

  std::string fn;
  int fd;

  Entry *db;
  unsigned int n;

  static void create(const char *_fn, unsigned int _n);

  IPDB(const char *_fn);
  ~IPDB();

  Entry *find(uint32_t ip);

  unsigned int burns_left(uint32_t ip);
  void use_burns(uint32_t ip, unsigned int burns);
};

#endif

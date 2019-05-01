#ifndef __MAKEMORE_MAPFILE_HH__
#define __MAKEMORE_MAPFILE_HH__ 1

#include <stdlib.h>
#include <stdio.h>

#include <sys/mman.h>

#include <string>
#include <map>

namespace makemore {

struct Mapfile {
  std::string fn;
  int fd;

  void *base;
  unsigned long size;
  unsigned long top;

  std::map<void *, std::pair<unsigned long, unsigned long> > ptrofflen;

  Mapfile(const std::string &_fn);
  ~Mapfile();

  template <class T> void map(T *cudata, unsigned long n) {
    mapv((void *)cudata, n * sizeof(T));
  }

  void mapv(void *cudata, unsigned long n);
  void grow(unsigned long new_size);

  void load(void *cudata);
  void load(void *cudata, unsigned long verify_len);
  void load();
  void save(void *cudata);
  void save(void *cudata, unsigned long verify_len);
  void save();
};

}

#endif

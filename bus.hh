#ifndef __MAKEMORE_BUS_HH__
#define __MAKEMORE_BUS_HH__ 1

#include <stdio.h>

#include <string>
#include <vector>

#include "parson.hh"
#include "pipeline.hh"

namespace makemore {

struct Bus {
  std::string fn;
  FILE *fp;
  int fd;
  bool locked;

  Bus();
  Bus(const std::string &_fn);
  ~Bus();

  void lock();
  void unlock();
  void open(const std::string &_fn);
  void close();

  void _seek_end();


  void add(const Parson *p, unsigned int n);
  void add(const Parson **pp, unsigned int n);

  void add(const Parson &x) {
    add(&x, 1);
  }
};

}

#endif

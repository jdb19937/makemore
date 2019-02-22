#ifndef __MAKEMORE_BUS_HH__
#define __MAKEMORE_BUS_HH__ 1

#include <vector>

#include "parson.hh"
#include "pipeline.hh"

namespace makemore {

struct Bus {
  unsigned int n;
  std::vector<Parson> seat;

  Bus();
  Bus(const char *fn);
  ~Bus();

  void load(const char *);
  void load(FILE *);
  void add(const Parson &);

  void save(FILE *);

  Parson *pick();
  Parson *pick(const char *tag, unsigned int max_tries);
  Parson *pick(const char *tag1, const char *tag2, unsigned int max_tries);

  void generate(Pipeline *pipe, long min_age = 0);
  void burn(Pipeline *pipe);
};

}

#endif

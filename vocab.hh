#ifndef __MAKEMORE_VOCAB_HH__
#define __MAKEMORE_VOCAB_HH__ 1

#include <assert.h>
#include <string.h>

#include "tagbag.hh"

#include <string>
#include <vector>

namespace makemore {

struct Vocab {
  Vocab();

  unsigned int n;
  std::vector<std::string> tags;
  std::vector<Tagbag> bags;

  void add(const char *tag);

  void encode(const char *str, Tagbag *tb);
  void decode(const Tagbag &tb, std::string *str);
};

}

#endif

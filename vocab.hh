#ifndef __MAKEMORE_VOCAB_HH__
#define __MAKEMORE_VOCAB_HH__ 1

#include <assert.h>
#include <string.h>

#include "tagbag.hh"

#include <string>
#include <vector>
#include <set>

namespace makemore {

struct Vocab {
  Vocab();

  unsigned int n;
  std::vector<char *> tags;
  std::vector<Tagbag> bags;

  std::set<std::string> seen_tag;

  void add(const char *tag);
  void add(const std::string &tagstr) {
    add(tagstr.c_str());
  }

  void decode(const Tagbag &tb, std::string *str);
};

}

#endif

#ifndef __MAKEMORE_VOCAB_HH__
#define __MAKEMORE_VOCAB_HH__ 1

#include <assert.h>
#include <string.h>

#include "hashbag.hh"

#include <string>
#include <vector>
#include <set>

namespace makemore {

struct Vocab {
  Vocab();

  unsigned int n;
  std::vector<char *> tags;
  std::vector<Hashbag> bags;

  std::set<std::string> seen_tag;

  void add(const char *tag);
  void add(const std::string &tagstr) {
    add(tagstr.c_str());
  }

  const char *closest(const Hashbag &x, const Hashbag **y) const;
};

}

#endif

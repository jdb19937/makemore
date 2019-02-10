#ifndef __MAKEMORE_VOCAB_HH__
#define __MAKEMORE_VOCAB_HH__ 1

#include <assert.h>
#include <string.h>

#include "tagbag.hh"

#include <string>
#include <vector>

namespace makemore {

struct Vocab {
  Vocab() {
    n = 0;
  }

  unsigned int n;
  std::vector<std::string> tags;
  std::vector<Tagbag> bags;

  void add(const char *tag);
  const char *decode(const Tagbag &tb);
};

}

#endif

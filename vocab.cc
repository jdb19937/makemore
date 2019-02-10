#define __MAKEMORE_VOCAB_CC__ 1
#include "vocab.hh"
#include "closest.hh"

namespace makemore {  

void Vocab::add(const char *tag) {
  tags.resize(n + 1);
  tags[n] = std::string(tag);

  bags.resize(n + 1);
  bags[n].add(tag);
}

const char *Vocab::decode(const Tagbag &tb) {
  const double *x = tb.vec;
  const double *m = (const double *)bags.data();
  unsigned int k = 256;

  unsigned int i = closest(x, m, k, n);
  assert(i >= 0 && i < n);

  return tags[i].c_str();
}
  
}

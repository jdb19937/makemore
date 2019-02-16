#ifndef __MAKEMORE_BRANE_HH__
#define __MAKEMORE_BRANE_HH__ 1

#include "shibboleth.hh"
#include "confab.hh"
#include "rule.hh"
#include "vocab.hh"

namespace makemore {

struct Brane {
  const unsigned int max_depth = 32;

  Vocab vocab;
  Confab *confab;

  Brane(Confab *_confab);
  ~Brane();

  void _init_vocab();
  Shibboleth ask(const Shibboleth &req, Shibboleth *memp = NULL, Shibboleth *auxp = NULL, unsigned int depth = 0);
  void burn(const Rule *rule, unsigned int mbn, double nu);
};

}

#endif

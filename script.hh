#ifndef __MAKEMORE_SCRIPT_HH__
#define __MAKEMORE_SCRIPT_HH__ 1

#include "shibboleth.hh"
#include "vocab.hh"
#include "rule.hh"

#include <string>
#include <map>

namespace makemore {

struct Script {
  std::vector<Rule> rules;

  std::string fn;
  FILE *fp;

  Script(const char *_fn, Vocab *vocab = NULL);
  ~Script();

  const Rule *pick();
};

}

#endif

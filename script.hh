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

  Script();
  ~Script();

  void load(FILE *fp);
  void load(const char *fn);

  const Rule *pick();
};

}

#endif

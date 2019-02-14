#ifndef __MAKEMORE_SCRIPT_HH__
#define __MAKEMORE_SCRIPT_HH__ 1

#include "shibboleth.hh"
#include "vocab.hh"

#include <string>
#include <map>

namespace makemore {

struct Script {
  typedef std::pair<std::string, std::string> Template;
  std::vector<Template> templates;

  std::multimap<std::string, std::string> defines;

  std::string fn;
  FILE *fp;

  Script(const char *_fn, Vocab *vocab = NULL);
  ~Script();

  void pick(Shibboleth *req, Shibboleth *rsp);
};

}

#endif

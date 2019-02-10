#ifndef __MAKEMORE_SCRIPT_HH__
#define __MAKEMORE_SCRIPT_HH__ 1

#include "tagbag.hh"
#include "vocab.hh"

namespace makemore {

struct Script {
  typedef std::pair<std::string, std::string> Template;
  std::vector<Template> templates;

  std::string fn;
  FILE *fp;

  Script(const char *_fn);
  ~Script();
};

}

#endif

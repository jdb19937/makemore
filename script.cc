#define __MAKEMORE_SCRIPT_CC__ 1

#include <assert.h>

#include "script.hh"
#include "strutils.hh"

namespace makemore {
using namespace std;

Script::Script(const char *_fn, Vocab *vocab) {
  fn = _fn;
  assert(fp = fopen(fn.c_str(), "r"));

  char buf[4096];
  while (1) {
    *buf = 0;
    char *unused = fgets(buf, sizeof(buf) - 1, fp);
    buf[sizeof(buf) - 1] = 0;
    char *p = strchr(buf, '\n');
    if (!p)
      break;
    *p = 0;

    p = strchr(buf, '#');
    if (p)
      *p = 0;
    const char *q = buf;
    while (*q == ' ')
      ++q;
    if (!*q)
      continue;

    if (vocab)
      vocab->add(q);

    Rule r;
    unsigned int copies = r.parse(q);
    for (unsigned int copy = 0; copy < copies; ++copy)
      rules.push_back(r);
  }

  fclose(fp);
  fp = NULL;
}

Script::~Script() {
}

const Rule *Script::pick() {
  assert(rules.size());
  return &rules[randuint() % rules.size()];
}

}

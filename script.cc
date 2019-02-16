#define __MAKEMORE_SCRIPT_CC__ 1

#include <assert.h>

#include "script.hh"
#include "strutils.hh"

namespace makemore {
using namespace std;

Script::Script() {

}

void Script::load(const char *fn) {
  FILE *fp;
  assert(fp = fopen(fn, "r"));
  load(fp);
  fclose(fp);
}

void Script::load(FILE *fp) {
  while (1) {
    int c = getc(fp);
    if (c == EOF)
      break;
    ungetc(c, fp);
   
    Rule r;
    r.load(fp);
    for (unsigned int copy = 0; copy < r.multiplicity; ++copy)
      rules.push_back(r);
  }
}

Script::~Script() {
}

const Rule *Script::pick() {
  assert(rules.size());
  return &rules[randuint() % rules.size()];
}

}

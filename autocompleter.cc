#define __MAKEMORE_AUTOCOMPLETER_CC__ 1
#include "autocompleter.hh"

#include <stdio.h>
#include <stdlib.h>

using namespace makemore;

#if MAIN
int main() {
  Autocompleter ac;
  ac.add("hello");
  ac.add("hello");
  ac.add("howdy");
  ac.add("there");

  std::vector<std::string> r;
  ac.find("h", &r);
  for (auto x : r) {
    printf("%s\n", x.c_str());
  }
  return 0;
}
#endif

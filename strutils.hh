#ifndef __MAKEMORE_STRUTILS_HH__
#define __MAKEMORE_STRUTILS_HH__ 1

#include <vector>
#include <string>

namespace makemore {

extern void split(const char *str, char sep, std::vector<std::string> *words);

inline std::string join(const std::vector<std::string> &v, const char *sep) {
  std::string out;
  for (auto i = v.begin(); i != v.end(); ++i) {
    if (out.length())
      out += sep;
    out += *i;
  }
  return out;
}

inline std::string join(const std::vector<std::string> &v, char sep) {
  char buf[2] = {sep, 0};
  return join(v, buf);
}

}

#endif

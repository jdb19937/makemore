#ifndef __MAKEMORE_STRUTILS_HH__
#define __MAKEMORE_STRUTILS_HH__ 1

#include <vector>
#include <string>
#include <map>

namespace makemore {

extern void split(const char *str, char sep, std::vector<std::string> *words);

extern void splitparts(const std::string &str, std::vector<std::string> *parts);
extern void splitwords(const std::string &str, std::vector<std::string> *words);
extern void splitlines(const std::string &str, std::vector<std::string> *lines);


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

inline std::string joinwords(const std::vector<std::string> &v) {
  std::string out;
  for (auto i = v.begin(); i != v.end(); ++i) {
    if (*i->c_str()) {
      if (out.length())
        out += " ";
      out += *i;
    }
  }
  return out;
}

bool read_line(FILE *, std::string *);

extern std::string varsubst(const std::string &str, const std::map<std::string, std::string>& dict);

extern std::string refsubst(const std::string &rsp, const std::string &req);

extern void jointhread(const std::vector<std::vector<std::string> > &thread, std::vector<std::string> *wordsp, const std::string &sep);
extern void splitthread(const std::vector<std::string> &words, std::vector<std::vector<std::string> > *threadp, const std::string &sep);

}

#endif

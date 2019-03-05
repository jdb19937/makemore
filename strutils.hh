#ifndef __MAKEMORE_STRUTILS_HH__
#define __MAKEMORE_STRUTILS_HH__ 1

#include <vector>
#include <string>
#include <map>

#include <assert.h>
#include <ctype.h>

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


inline void slurp(FILE *fp, std::string *str) {
  char buf[1024];
  size_t ret;

  *str = "";

  while (1) {
    ret = fread(buf, 1, 1024, fp);
    if (ret < 1)
      return;
    *str += std::string(buf, ret);
  }
}

inline std::string slurp(FILE *fp) {
  std::string ret;
  slurp(fp, &ret);
  return ret;
}


inline void slurp(const std::string &fn, std::string *str) {
  FILE *fp;
  assert(fp = fopen(fn.c_str(), "r"));
  slurp(fp, str);
  fclose(fp);
}

inline std::string slurp(const std::string &fn) {
  std::string ret;
  slurp(fn, &ret);
  return ret;
}

bool read_line(FILE *, std::string *);

extern std::string varsubst(const std::string &str, const std::map<std::string, std::string>& dict);

extern std::string refsubst(const std::string &rsp, const std::string &req);

extern void jointhread(const std::vector<std::vector<std::string> > &thread, std::vector<std::string> *wordsp, const std::string &sep);
extern void splitthread(const std::vector<std::string> &words, std::vector<std::vector<std::string> > *threadp, const std::string &sep);

inline std::string to_hex(const std::string &binstr) {
  unsigned int n = binstr.length();
  const uint8_t *bin = (const uint8_t *)binstr.data();
  const char *tab = "0123456789ABCDEF";
  std::string hexstr;
  hexstr.resize(n * 2);
  for (unsigned int i = 0, j = 0; i < n; ++i, j += 2) {
    hexstr[j + 0] = tab[bin[i] >> 4];
    hexstr[j + 1] = tab[bin[i] & 0xF];
  }
  return hexstr;
}

inline std::string lowercase(std::string str) {
  unsigned int n = str.length();
  std::string lcstr;
  lcstr.resize(n);
  for (unsigned int i = 0; i < n; ++i)
    lcstr[i] = tolower(str[i]);
  return lcstr;
}

}

#endif

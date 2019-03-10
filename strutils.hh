#ifndef __MAKEMORE_STRUTILS_HH__
#define __MAKEMORE_STRUTILS_HH__ 1

#include <vector>
#include <string>
#include <map>

#include <assert.h>
#include <ctype.h>

namespace makemore {

typedef std::vector<std::string> strvec;
typedef std::vector<strvec> strmat;

extern void split(const char *str, char sep, strvec *words);

extern void splitparts(const std::string &str, strvec *parts);
extern void splitwords(const std::string &str, strvec *words);
extern void splitlines(const std::string &str, strvec *lines);


inline std::string join(const strvec &v, const char *sep) {
  std::string out;
  for (auto i = v.begin(); i != v.end(); ++i) {
    if (out.length())
      out += sep;
    out += *i;
  }
  return out;
}

inline std::string join(const strvec &v, char sep) {
  char buf[2] = {sep, 0};
  return join(v, buf);
}

inline std::string joinwords(const strvec &v) {
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

extern void jointhread(const strmat &thread, strvec *wordsp, const std::string &sep);
extern void splitthread(const strvec &words, strmat *threadp, const std::string &sep);

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

inline bool hasspace(const std::string &s) {
  for (const char *p = s.c_str(); *p; ++p) {
    if (isspace(*p))
      return true;
  }
  return false;
}

inline void catstrvec(strvec &a, const strvec &b) {
  unsigned long as = a.size();
  unsigned long bs = b.size();
  unsigned long cs = as + bs;
  a.resize(cs);
  for (unsigned int i = as; i < cs; ++i)
    a[i] = b[i - as];
}

}

#endif

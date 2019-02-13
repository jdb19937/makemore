#ifndef __MAKEMORE_SHIBBOLETH_HH__
#define __MAKEMORE_SHIBBOLETH_HH__ 1

#include <assert.h>
#include <string.h>

#include <string>

#include "random.hh"
#include "wordvec.hh"

namespace makemore {

struct Shibboleth {
  const double omul = 0.1;

  Wordvec avec, ovec;
  unsigned int wn;

  void clear() {
    wn = 0;
    avec.clear();
    ovec.clear();
  }

  Shibboleth() {
    clear();
  }

  void copy(const Shibboleth &t) {
    wn = t.wn;
    avec = t.avec;
    ovec = t.ovec;
  }

  Shibboleth(const Shibboleth &t) {
    copy(t);
  }

  Shibboleth &operator =(const Shibboleth &tb) {
    copy(tb);
    return *this;
  }


  void unshift(const char *);
  void unshift(const std::string &x) { unshift(x.c_str()); }
  void push(const char *);
  void push(const std::string &x) { push(x.c_str()); }

  void encode(const char *str, unsigned int seed = 0);

  void encode(const std::string &str, unsigned int seed = 0) {
    encode(str.c_str(), seed);
  }

  std::string decode(const class Vocab &vocab);
};

}

#endif

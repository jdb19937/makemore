#ifndef __MAKEMORE_SHIBBOLETH_HH__
#define __MAKEMORE_SHIBBOLETH_HH__ 1

#include <assert.h>
#include <string.h>

#include <string>
#include <map>

#include "random.hh"
#include "vocab.hh"
#include "hashbag.hh"

namespace makemore {

struct Shibboleth {
  const double omul = 0.2;

  Hashbag avec, ovec;
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

  void encode(const char *str, class Vocab *vocab = NULL, unsigned int seed = 0, std::multimap<std::string, std::string> *defines = NULL, std::map<std::string, std::string> *assign = NULL);

  std::string decode(const class Vocab &vocab);
};

}

#endif

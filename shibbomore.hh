#ifndef __MAKEMORE_SHIBBOMORE_HH__
#define __MAKEMORE_SHIBBOMORE_HH__ 1

#include <assert.h>
#include <string.h>

#include <string>
#include <map>
#include <algorithm>

#include "random.hh"
#include "vocab.hh"
#include "hashbag.hh"
#include "shibboleth.hh"

namespace makemore {

struct Shibbomore {
  Hashbag front[3];
  Shibboleth backleth;

  void clear() {
    front[0].clear();
    front[1].clear();
    front[2].clear();
    backleth.clear();
  }

  bool is_empty() const {
    return (front[0].size() < 0.5);
  }

  double size() const {
    if (front[0].size() < 0.5)
      return 0.0;

    double n = 1.0;
    if (front[1].size() >= 0.5)
      n += 1.0;
    if (front[2].size() >= 0.5)
      n += 1.0;
    n += backleth.size();

    return n;
  }

  Shibbomore() {
    clear();
  }

  void add(const Shibbomore &t) {
    front[0].add(t.front[0]);
    front[1].add(t.front[1]);
    front[2].add(t.front[2]);
    backleth.add(t.backleth);
  }

  void mul(double m) {
    front[0].mul(m);
    front[1].mul(m);
    front[2].mul(m);
    backleth.mul(m);
  }

  void copy(const Shibbomore &t) {
    front[0] = t.front[0];
    front[1] = t.front[1];
    front[2] = t.front[2];
    backleth = t.backleth;
  }

  Shibbomore(const Shibbomore &t) {
    copy(t);
  }

  Shibbomore &operator =(const Shibbomore &tb) {
    copy(tb);
    return *this;
  }

  void encode(const char *str);
  void encode(const std::string &str) {
    encode(str.c_str());
  }
  void encode(const std::vector<std::string>& vec);

  std::string decode(const class Vocab &vocab) const;

  void save(FILE *fp) const;
  void load(FILE *fp);
};

}

#endif

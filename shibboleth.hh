#ifndef __MAKEMORE_SHIBBOLETH_HH__
#define __MAKEMORE_SHIBBOLETH_HH__ 1

#include <assert.h>
#include <string.h>

#include <string>
#include <map>
#include <algorithm>

#include "random.hh"
#include "vocab.hh"
#include "hashbag.hh"

namespace makemore {

struct Shibboleth {
  Hashbag head, torso, rear, pairs;

  void clear() {
    head.clear();
    torso.clear();
    rear.clear();
    pairs.clear();
  }

  bool is_empty() const {
    return (head.size() <= 0.5);
  }

  bool is_single() const {
    return (rear.size() <= 0.5);
  }

  double size() const {
    double n = 0;
    n += (head.size() > 0.5) ? 1.0 : 0.0;
    n += (rear.size() > 0.5) ? 1.0 : 0.0;
    n += torso.size();
    return n;
  }

  Shibboleth() {
    clear();
  }

  void add(const Shibboleth &t) {
    head.add(t.head);
    torso.add(t.torso);
    pairs.add(t.pairs);
    rear.add(t.rear);
  }

  void mul(double m) {
    head.mul(m);
    torso.mul(m);
    pairs.mul(m);
    rear.mul(m);
  }

  void copy(const Shibboleth &t) {
    head = t.head;
    torso = t.torso;
    pairs = t.pairs;
    rear = t.rear;
  }

  void copy(const Hashbag &h) {
    head = h;
    torso.clear();
    rear.clear();
    pairs.clear();
  }
  
  Shibboleth(const Shibboleth &t) {
    copy(t);
  }

  Shibboleth &operator =(const Shibboleth &tb) {
    copy(tb);
    return *this;
  }

  void reverse() {
    std::swap(head, rear);
  }

  void negate() {
    head.negate();
    rear.negate();
    torso.negate();
    // don't negate pairs
  }

  void append(const char *);
  void append(const Shibboleth &x);
  void append(const Hashbag &);

  void prepend(const char *);
  void prepend(const Hashbag &);
  void prepend(const Shibboleth &x);

  void encode(const std::string &str) {
    encode(str.c_str());
  }
  void encode(const char *str);
  void encode(const std::vector<std::string> &vec);

  std::string decode(const class Vocab &vocab, bool force = false) const;

  void save(FILE *fp) const;
  void load(FILE *fp);
};

}

#endif

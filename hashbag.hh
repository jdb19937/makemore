#ifndef __MAKEMORE_HASHBAG_HH__
#define __MAKEMORE_HASHBAG_HH__ 1

#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include <string>

#include "random.hh"

namespace makemore {

struct Hashbag {
  const static unsigned int n = 256;
  double vec[n];

  void clear() {
    memset(vec, 0, sizeof(vec));
  }

  double size() const {
    double e2 = 0;
    for (unsigned int i = 0; i < n; ++i)
      e2 += (vec[i] * vec[i]);
    e2 /= (double)n;
    return e2;
  }

  double sum() const {
    double s = 0;
    for (unsigned int i = 0; i < n; ++i)
      s += vec[i];
    return s;
  }

  Hashbag() {
    clear();
  }

  Hashbag(const Hashbag &t) {
    copy(t);
  }

  Hashbag(const char *tag) {
    clear();
    add(tag);
  }

  static Hashbag random() {
    char buf[32];
    sprintf(buf, "%08x", randuint());
    return Hashbag(buf);
  }


  void add(const char *tag, double m = 1.0);

  void copy(const Hashbag &t) {
    memcpy(vec, t.vec, sizeof(double) * n);
  }

  void mul(const Hashbag &x) {
    for (unsigned int i = 0; i < n; ++i)
      vec[i] *= x.vec[i];
  }
  
  void mul(double w) {
    if (w == 1.0)
      return;
    for (unsigned int i = 0; i < n; ++i)
      vec[i] *= w;
  }

  Hashbag &operator *= (const Hashbag &x) {
    mul(x);
    return *this;
  }

  Hashbag &operator *= (double w) {
    mul(w);
    return *this;
  }

  Hashbag operator * (const Hashbag &x) const {
    Hashbag tb = *this;
    return (tb *= x);
  }

  Hashbag operator * (double w) const {
    Hashbag tb = *this;
    return (tb *= w);
  }

  void add(const Hashbag &tb) {
    for (unsigned int i = 0; i < n; ++i) {
       vec[i] += tb.vec[i];
    }
  }

  void sub(const Hashbag &tb) {
    for (unsigned int i = 0; i < n; ++i) {
       vec[i] -= tb.vec[i];
    }
  }

  Hashbag &operator += (const Hashbag &tb) {
    add(tb);
    return *this;
  }

  Hashbag &operator -= (const Hashbag &tb) {
    sub(tb);
    return *this;
  }

  Hashbag operator + (const Hashbag &tb1) const {
    Hashbag tb0 = *this;
    tb0.add(tb1);
    return tb0;
  }

  Hashbag operator - (const Hashbag &tb1) const {
    Hashbag tb0 = *this;
    tb0.sub(tb1);
    return tb0;
  }


  void clamp() {
    for (unsigned int i = 0; i < n; ++i) {
      vec[i] = sigmoid(vec[i]);
    }
  }

  void unclamp() {
    for (unsigned int i = 0; i < n; ++i) {
      assert(vec[i] > 0 && vec[i] < 1);
      vec[i] = unsigmoid(vec[i]);
    }
  }

  Hashbag &operator =(const Hashbag &tb) {
    copy(tb);
    return *this;
  }

  void negate() {
    for (unsigned int i = 0; i < n; ++i)
      vec[i] = -vec[i];
  }

  void save(FILE *fp) const {
    size_t ret;
    ret = fwrite(vec, sizeof(double), n, fp);
    assert(ret == n);
  }

  void load(FILE *fp) {
    size_t ret;
    ret = fread(vec, sizeof(double), n, fp);
    assert(ret == n);
  }

  std::string guesstract(const class Vocab &v, double nfloor = 0.125);
};

}

#endif

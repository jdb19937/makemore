#ifndef __MAKEMORE_HASHBAG_HH__
#define __MAKEMORE_HASHBAG_HH__ 1

#include <assert.h>
#include <string.h>

#include <string>

#include "random.hh"

namespace makemore {

struct Hashbag {
  const static unsigned int n = 256;
  double vec[n];

  void clear() {
    memset(vec, 0, sizeof(vec));
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

  void add(const char *tag, double m = 1.0);

  void copy(const Hashbag &t) {
    memcpy(vec, t.vec, sizeof(vec));
  }

  void mul(double w) {
    if (w == 1.0)
      return;
    for (unsigned int i = 0; i < n; ++i)
      vec[i] *= w;
  }
  

  Hashbag &operator *= (double w) {
    mul(w);
    return *this;
  }

  Hashbag operator * (double w) {
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

  Hashbag operator + (const Hashbag &tb1) {
    Hashbag tb0 = *this;
    tb0.add(tb1);
    return tb0;
  }

  Hashbag operator - (const Hashbag &tb1) {
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

  double abs() {
    double e = 0;
    for (unsigned int i = 0; i < n; ++i)
      e += vec[i] * vec[i];
    return e;
  }
};

}

#endif

#ifndef __MAKEMORE_WORDVEC_HH__
#define __MAKEMORE_WORDVEC_HH__ 1

#include <assert.h>
#include <string.h>

#include <string>

#include "random.hh"

namespace makemore {

struct Wordvec {
  const static unsigned int n = 256;
  double vec[n];

  void clear() {
    memset(vec, 0, sizeof(vec));
  }

  Wordvec() {
    clear();
  }

  Wordvec(const Wordvec &t) {
    copy(t);
  }

  Wordvec(const char *tag) {
    clear();
    add(tag);
  }

  void add(const char *tag, double m = 1.0);

  void copy(const Wordvec &t) {
    memcpy(vec, t.vec, sizeof(vec));
  }

  void mul(double w) {
    if (w == 1.0)
      return;
    for (unsigned int i = 0; i < n; ++i)
      vec[i] *= w;
  }
  

  Wordvec &operator *= (double w) {
    mul(w);
    return *this;
  }

  Wordvec operator * (double w) {
    Wordvec tb = *this;
    return (tb *= w);
  }

  void add(const Wordvec &tb) {
    for (unsigned int i = 0; i < n; ++i) {
       vec[i] += tb.vec[i];
    }
  }

  void sub(const Wordvec &tb) {
    for (unsigned int i = 0; i < n; ++i) {
       vec[i] -= tb.vec[i];
    }
  }

  Wordvec &operator += (const Wordvec &tb) {
    add(tb);
    return *this;
  }

  Wordvec &operator -= (const Wordvec &tb) {
    sub(tb);
    return *this;
  }

  Wordvec operator + (const Wordvec &tb1) {
    Wordvec tb0 = *this;
    tb0.add(tb1);
    return tb0;
  }

  Wordvec operator - (const Wordvec &tb1) {
    Wordvec tb0 = *this;
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

  Wordvec &operator =(const Wordvec &tb) {
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

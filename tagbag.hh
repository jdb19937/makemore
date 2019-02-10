#ifndef __MAKEMORE_TAGBAG_HH__
#define __MAKEMORE_TAGBAG_HH__ 1

#include <assert.h>
#include <string.h>

#include "random.hh"

namespace makemore {

struct Tagbag {
  const static unsigned int n = 256;
  double vec[n];

  void clear() {
    memset(vec, 0, sizeof(vec));
  }

  Tagbag() {
    clear();
  }

  void copy(const Tagbag &t) {
    memcpy(vec, t.vec, sizeof(vec));
  }

  Tagbag(const Tagbag &t) {
    copy(t);
  }

  Tagbag(const Tagbag &t, double w) {
    copy(t);
    mul(w);
  }

  Tagbag(const char *tag, double w = 1);

  void mul(double w) {
    if (w == 1.0)
      return;
    for (unsigned int i = 0; i < n; ++i)
      vec[i] *= w;
  }
  

  Tagbag &operator *= (double w) {
    mul(w);
    return *this;
  }

  Tagbag operator * (double w) {
    Tagbag tb = *this;
    return (tb *= w);
  }

  void add(const Tagbag &tb) {
    for (unsigned int i = 0; i < n; ++i) {
       vec[i] += tb.vec[i];
    }
  }

  void sub(const Tagbag &tb) {
    for (unsigned int i = 0; i < n; ++i) {
       vec[i] -= tb.vec[i];
    }
  }

  Tagbag &operator += (const Tagbag &tb) {
    add(tb);
    return *this;
  }

  Tagbag &operator -= (const Tagbag &tb) {
    sub(tb);
    return *this;
  }

  Tagbag operator + (const Tagbag &tb1) {
    Tagbag tb0 = *this;
    tb0.add(tb1);
    return tb0;
  }

  Tagbag operator - (const Tagbag &tb1) {
    Tagbag tb0 = *this;
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

  Tagbag &operator =(const Tagbag &tb) {
    copy(tb);
    return *this;
  }
};

}

#endif

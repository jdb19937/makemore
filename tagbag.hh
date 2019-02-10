#ifndef __MAKEMORE_TAGBAG_HH__
#define __MAKEMORE_TAGBAG_HH__ 1

#include <assert.h>
#include <string.h>

#include "random.hh"

namespace makemore {

struct Tagbag {
  const static unsigned int n = 256;
  double vec[n];

  Tagbag() {
    memset(vec, 0, sizeof(vec));
  }

  Tagbag(const Tagbag &t) {
    memcpy(vec, t.vec, sizeof(vec));
  }

  Tagbag(const char *tag);

  Tagbag &operator *= (double w) {
    for (unsigned int i = 0; i < n; ++i) {
       vec[i] *= w;
    }
    return *this;
  }

  Tagbag operator * (double w) {
    Tagbag tb = *this;
    return (tb *= w);
  }

  void add(const char *tag, double w) {
    Tagbag tb(tag);
    tb *= w;
    *this += tb;
  }

  Tagbag &operator += (const Tagbag &t) {
    for (unsigned int i = 0; i < n; ++i) {
       vec[i] += t.vec[i];
    }
    return *this;
  }

  Tagbag operator + (const Tagbag &tb1) {
    Tagbag tb0 = *this;
    return (tb0 += tb1);
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
};

}

#endif

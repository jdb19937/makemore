#ifndef __MAKEMORE_LINEREF_HH__
#define __MAKEMORE_LINEREF_HH__ 1

#include <stdio.h>

#include "strutils.hh"
#include "cudamem.hh"

namespace makemore {

struct Lineref {
  typedef enum {
    TYPE_NONE,
    TYPE_STRVEC,
    TYPE_CUDA
  } Type;

  Type type;
  unsigned int refs;

  union {
    struct {
      cudouble *ptr;
      unsigned int len;
    } _as_cuda;

    strvec _as_strvec;
  };

  Lineref() {
    type = TYPE_NONE;
    refs = 0;
  }

  Lineref(Type _type) {
    type = _type;
    if (type == TYPE_STRVEC)
      new ( strvec_ptr() ) strvec();
    refs = 0;
  }

  ~Lineref() {
    if (type == TYPE_STRVEC)
      strvec_ptr()->~strvec();
  }

  strvec *strvec_ptr() {
    return ((strvec *)&_as_strvec);
  }

  cudouble *cuda_ptr() {
    return ((cudouble *)_as_cuda.ptr);
  }
};

}

#endif

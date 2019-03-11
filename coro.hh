#ifndef __MAKEMORE_CORO_HH__
#define __MAKEMORE_CORO_HH__ 1

#include <stdio.h>
#include <unistd.h>
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <ucontext.h>

namespace makemore {

typedef void (*corofunc_t)(...);

template <class T, unsigned int SS = 65536> struct Coro {
  uint8_t stack[SS];
  ucontext_t me;
  ucontext_t uc;
  T retval;
  volatile bool done;
  volatile bool inuc;

  Coro(corofunc_t f, void *arg) {
    inuc = false;
    done = false;
    ::getcontext(&uc);

    uc.uc_stack.ss_sp = stack;
    uc.uc_stack.ss_size = SS;
    uc.uc_stack.ss_flags = 0;
    uc.uc_link = 0;

    ::makecontext(&uc, (void (*)())f, 2, arg, NULL);
  }

  ~Coro() {
  }

  T* operator()() {
    return get();
  }

  T* get() {
    if (done)
      return NULL;

    assert(!inuc);
    inuc = true;
    ::swapcontext(&me, &uc);

    assert(!inuc);

    if (done)
      return NULL;
    return &retval;
  };

  void yield(const T &x) {
    assert(!done);
    assert(inuc);

    retval = x;
    inuc = false;
    ::swapcontext(&uc, &me);

    assert(inuc);
  }

  void finish() {
    assert(!done);
    assert(inuc);

    done = true;
    inuc = false;
    ::setcontext(&me);
    assert(0);
  }
};

}

#endif



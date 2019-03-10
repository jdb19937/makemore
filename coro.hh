#ifndef __MAKEMORE_CORO_HH__
#define __MAKEMORE_CORO_HH__ 1

#include <stdio.h>
#include <unistd.h>
#include <assert.h>
#include <stdlib.h>
#include <ucontext.h>

namespace makemore {

template <class T> struct Coro {
  typedef void (*Signature)(Coro<T> *);

  char *stack;
  ucontext_t me;
  ucontext_t uc;
  T retval;
  bool done;

  Coro(Signature f, unsigned int ss = 4096) {
    done = false;
    ::getcontext(&uc);

    stack = new char[ss];
    uc.uc_stack.ss_sp = stack;
    uc.uc_stack.ss_size = ss;
    uc.uc_stack.ss_flags = 0;

    ::makecontext(&uc, (void (*)())f, 1, this);
  }

  ~Coro() {
    delete[] stack;
  }

  T* operator()() {
    if (done)
      return NULL;

    int restore = 0;
    ::getcontext(&me);
    if (restore) {
      if (done)
        return NULL;
      return &retval;
    }
    restore = 1;

    ::setcontext(&uc);
    assert(0);
  };

  void yield(const T &x) {
    assert(!done);

    int stop = 1;
    ::getcontext(&uc);
    if (stop) {
      stop = 0;
      retval = x;
      ::setcontext(&me);
    }
  }

  void yield() {
    assert(!done);

    done = true;
    ::setcontext(&me);
  }
};

}

#endif



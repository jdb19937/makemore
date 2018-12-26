#ifndef __MAKEMORE_PERSIST_HH__
#define __MAKEMORE_PERSIST_HH__ 1

#include <stdio.h>

struct Persist {
  virtual void load(FILE *) = 0;
  virtual void save(FILE *) = 0;

  void load(const char *fn);
  void save(const char *fn);
};

template <class T> inline T *load_new(FILE *fp) {
  Persist *x = new T();
  x->load(fp);
  return x;
}

template <class T> inline T *load_new(const char *fn) {
  Persist *x = new T();
  x->load(fn);
  return x;
}
  

#endif

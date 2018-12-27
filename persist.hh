#ifndef __MAKEMORE_PERSIST_HH__
#define __MAKEMORE_PERSIST_HH__ 1

#include <stdio.h>

struct Persist {
  inline bool _check_eof(FILE *fp) {
    int c = getc(fp);
    if (c == EOF)
      return true;
    ungetc(c, fp);
    return false;
  }

  virtual void load(FILE *) = 0;
  virtual void save(FILE *) const = 0;

  virtual void load(const char *fn);
  virtual void save(const char *fn) const;
};

template <class T> inline T *load_new(FILE *fp) {
  Persist *x = new T();
  ((T *)x)->load(fp);
  return ((T *)x);
}

template <class T> inline T *load_new(const char *fn) {
  Persist *x = new T();
  ((T *)x)->load(fn);
  return ((T *)x);
}
  

#endif

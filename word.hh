#ifndef __MAKEMORE_WORD_HH__
#define __MAKEMORE_WORD_HH__ 1

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <string>
#include <vector>

#include "cudamem.hh"
#include "strutils.hh"

namespace makemore {

struct Word;

typedef std::vector<Word> Line;
typedef std::list<Line> Grid;

extern void line_to_strvec(const Line &wv, strvec *svp);
extern void strvec_to_line(const strvec &sv, Line *wvp);

struct Word {
  void *ptr;
  unsigned int len;
  unsigned int *refs;
  bool cuda;

  Word() {
    ptr = NULL;
    len = 0;
    cuda = false;
    refs = NULL;
  }

  Word(void *_ptr, unsigned int _len, bool _cuda = false) {
    ptr = _ptr;
    len = _len;
    cuda = _cuda;
    refs = new unsigned int(1);
  }

  Word(const std::string &s) {
    len = s.length();
    ptr = new uint8_t[len + 1];
    memcpy((uint8_t *)ptr, (const uint8_t *)s.data(), len);
    ((uint8_t *)ptr)[len] = 0;
    cuda = false;
    refs = new unsigned int(1);
  }

  void copy(const std::string &s) {
    clear();
    len = s.length();
    ptr = new uint8_t[len + 1];
    memcpy((uint8_t *)ptr, (const uint8_t *)s.data(), len);
    ((uint8_t *)ptr)[len] = 0;
    cuda = false;
    refs = new unsigned int(1);
  }

  void copy(void *_ptr, unsigned int _len, bool _cuda = false) {
    clear();
    ptr = _ptr;
    len = _len;
    cuda = _cuda;
    refs = new unsigned int(1);
  }

  Word(const Word &w) {
    ptr = w.ptr;
    len = w.len;
    cuda = w.cuda;
    refs = w.refs;

    if (refs) {
      assert(*refs);
      ++*refs;
    } else {
      assert(!ptr);
    }
  }

  void copy(const Word &w) {
    clear();

    ptr = w.ptr;
    len = w.len;
    cuda = w.cuda;
    refs = w.refs;

    if (refs) {
      assert(*refs);
      ++*refs;
    } else {
      assert(!ptr);
    }
  }

  Word &operator = (const Word &w) {
    copy(w);
    return *this;
  }

  Word &operator = (const std::string &s) {
    copy(s);
    return *this;
  }

  void clear() {
    if (!ptr) {
      assert(!refs);
      return;
    }

    assert(refs);
    assert(*refs);
    if (*refs > 1) {
      --*refs;
      refs = NULL;
      ptr = NULL;
      return;
    }
    delete refs;
    refs = NULL;

    if (cuda) {
      cufreev(ptr);
    } else {
      delete[] ((uint8_t *)ptr);
    }

    ptr = NULL;
  }
  
  ~Word() {
    if (!ptr) {
      assert(!refs);
      return;
    }

    assert(refs);
    assert(*refs);
    if (*refs > 1) {
      --*refs;
      return;
    }
    delete refs;

    if (cuda) {
      cufreev(ptr);
    } else {
      delete[] ((uint8_t *)ptr);
    }
  }

  void cudify() {
    if (cuda)
      return;
    uint8_t *new_ptr;
    makemore::cumake(&new_ptr, len);
    makemore::encude((const uint8_t *)ptr, len, new_ptr);
    copy(new_ptr, len, true);
  }

  operator std::string() {
    if (cuda) {
      std::string ret;
      ret.resize(len);
      makemore::decude((const uint8_t *)ptr, len, (uint8_t *)ret.data());
      return ret;
    } else {
      return std::string((char *)ptr, len);
    }
  }
};

}

#endif

#ifndef __MAKEMORE_IO_HH__
#define __MAKEMORE_IO_HH__ 1

#include <assert.h>
#include <stdlib.h>

#include <string>
#include <vector>
#include <list>

#include "strutils.hh"
#include "command.hh"
#include "word.hh"

namespace makemore {

class Session;
class Process;

struct IO {
  std::list<Line*> q;
  unsigned int n, l, m;

  std::list<Session*> session_readers, session_writers;
  std::list<Process*> process_readers, process_writers;

  unsigned int nreaders;
  unsigned int nwriters;

  IO() {
    m = 64;
    l = 16;
    n = 0;

    nreaders = 0;
    nwriters = 0;
fprintf(stderr, "new io %ld\n", (long)this);
  }

  ~IO() {
fprintf(stderr, "del io %ld\n", (long)this);
    assert(session_readers.begin() == session_readers.end());
    assert(session_writers.begin() == session_writers.end());
    assert(process_readers.begin() == process_readers.end());
    assert(process_writers.begin() == process_writers.end());
    assert(nreaders == 0);
    assert(nwriters == 0);

    for (auto wv : q) {
      delete wv;
    }
  }

  void autodestruct() {
    if (nreaders == 0 && nwriters == 0)
      delete this;
  }

  bool should_wake() const {
    return (n < l && nreaders);
  }
  bool can_put() const {
    return (n < m && nreaders);
  }
  bool done_put() const {
    return (nreaders == 0);
  }
  bool put(Line *);
  bool put(const strvec &);

  bool can_get() const {
    return (n > 0);
  }
  bool done_get() const {
    return (n == 0 && nwriters == 0);
  }
  Line *get();
  Line *peek();

  void wake_readers();
  void wake_writers();

  void link_reader(Session *reader);
  void link_reader(Process *reader);
  void unlink_reader(Session *reader);
  void unlink_reader(Process *reader);

  void link_writer(Session *writer);
  void link_writer(Process *writer);
  void unlink_writer(Session *writer);
  void unlink_writer(Process *writer);
};

}

#endif

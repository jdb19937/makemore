#ifndef __MAKEMORE_IO_HH__
#define __MAKEMORE_IO_HH__ 1

#include <assert.h>
#include <stdlib.h>

#include <string>
#include <vector>
#include <list>

#include "strutils.hh"
#include "command.hh"

namespace makemore {

class Session;
class Process;

struct IO {
  std::list<strvec> q;
  unsigned int n, l, m;
  strvec x;

  Session *head_session_reader;
  Session *head_session_writer;
  Process *head_process_reader;
  Process *head_process_writer;
  unsigned int nreaders;
  unsigned int nwriters;

  IO() {
    m = 64;
    l = 16;
    n = 0;

    nreaders = 0;
    nwriters = 0;

    head_session_reader = NULL;
    head_session_writer = NULL;
    head_process_reader = NULL;
    head_process_writer = NULL;
  }

  ~IO() {
    assert(!head_session_reader);
    assert(!head_session_writer);
    assert(!head_process_reader);
    assert(!head_process_writer);
    assert(nreaders == 0);
    assert(nwriters == 0);
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
  bool put(const strvec &);

  bool can_get() const {
    return (n > 0);
  }
  bool done_get() const {
    return (n == 0 && nwriters == 0);
  }
  strvec *get();
  strvec *peek();

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

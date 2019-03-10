#ifndef __MAKEMORE_PROCESS_HH__
#define __MAKEMORE_PROCESS_HH__ 1

#include <assert.h>

#include <string>
#include <vector>
#include <list>

#include "strutils.hh"
#include "command.hh"

namespace makemore {

struct Process {
  class Server *server;
  class Urbite *who;

  bool done;
  std::list<strvec> inq;
  unsigned int inqn, inqm;

  void *state;

  typedef enum {
    OUTPUT_TO_NULL,
    OUTPUT_TO_PROCESS,
    OUTPUT_TO_AGENT
  } OutputType;

  OutputType out_type;
    
  union OutputHandle {
    Process *process;
    class Agent *agent;
  };

  OutputHandle out;

  std::list<Process*> process_refs;

  void add_process_ref(Process *ref) {
    process_refs.push_back(ref);
  }
  bool remove_process_ref(Process *process) {
    auto i = process_refs.begin(); 
    while (i != process_refs.end()) {
      if (*i == process) {
        process_refs.erase(i);
        return true;
      }
      ++i;
    }
    return false;
  }

  Command cmd;
  strvec pre;

  Process(
    Server *_server,
    const Urbite *_who,
    Command _cmd,
    const strvec &_pre,
    OutputType,
    OutputHandle
  );

  ~Process();
  void deref();

  bool run();

  const strvec &peek_in() const {
    return *inq.begin();
  }

  bool in_empty() const {
    return (inq.begin() == inq.end());
  }

  bool in_ready() const {
    return (inqn < inqm);
  }

  bool out_ready() const;

  void pop_in() {
    assert(inqn > 0);
    inq.erase(inq.begin());
    --inqn;
  }

  bool put_in(const strvec &invec, bool force = false) {
    if (!force && !in_ready())
      return false;

    ++inqn;
    inq.push_back(invec);
    return true;
  }

  bool put_out(const strvec &outvec, bool force = false);
};

}

#endif

#ifndef __MAKEMORE_PROCESS_HH__
#define __MAKEMORE_PROCESS_HH__ 1

#include <assert.h>

#include <string>
#include <vector>
#include <list>

#include "strutils.hh"
#include "command.hh"
#include "coro.hh"

namespace makemore {

struct Process {
  class System *system;
  class Urbite *who;

  typedef enum {
    MODE_THINKING,
    MODE_READING,
    MODE_WRITING,
    MODE_DONE
  } Mode;

  Mode mode;
  Coro<Mode> *coro;

  std::list<strvec> inq;
  unsigned int inqn, inqm;
  strvec inx;

  Process *inproc, *outproc;
  class Agent *inagent, *outagent;

  Command cmd;
  strvec args;

  bool woke;
  struct Process *prev_proc, *next_proc;
  struct Process *prev_woke, *next_woke;
  struct Process *prev_done, *next_done;
  void wake();
  void sleep();
  void finish();

  Process(
    System *_system,
    const Urbite *_who,
    Command _cmd,
    const strvec &_pre,
    Process *inproc,
    Process *outproc,
    Agent *inagent,
    Agent *outagent
  );

  ~Process();

  bool can_put() const {
    return (inqn < inqm);
  }

  void put(const strvec &in);

  strvec *read();
  bool write(const strvec &outvec);

  bool run();
};

}

#endif

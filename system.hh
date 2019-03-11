#ifndef __MAKEMORE_SYSTEM_HH__
#define __MAKEMORE_SYSTEM_HH__ 1

namespace makemore {

struct Process;

struct System {
  Process *head_proc;
  Process *head_woke;
  Process *head_done;

  System();
  ~System();

  void run();
};

}

#endif

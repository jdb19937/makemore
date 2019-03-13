#ifndef __MAKEMORE_SYSTEM_HH__
#define __MAKEMORE_SYSTEM_HH__ 1

#include "process.hh"

namespace makemore {

struct Process;

struct System {
  struct Server *server;

  Process *head_proc;
  Process *head_woke;
  Process *head_done;

  System();
  ~System();

  void run();

  void link_done(Process *p) {
    p->prev_done = NULL;
    p->next_done = head_done;
    if (head_done) {
      assert(!head_done->prev_done);
      head_done->prev_done = p;
    }
    head_done = p;
  }

  void unlink_done(Process *p) {
    if (p->next_done)
      p->next_done->prev_done = p->prev_done;
    if (p->prev_done)
      p->prev_done->next_done = p->next_done;
    if (head_done == p)
      head_done = p->next_done;
    p->prev_done = NULL;
    p->next_done = NULL;
  }

  void link_woke(Process *p) {
    p->prev_woke = NULL;
    p->next_woke = head_woke;
    if (head_woke) {
      assert(!head_woke->prev_woke);
      head_woke->prev_woke = p;
    }
    head_woke = p;
  }

  void unlink_woke(Process *p) {
    if (p->next_woke)
      p->next_woke->prev_woke = p->prev_woke;
    if (p->prev_woke)
      p->prev_woke->next_woke = p->next_woke;
    if (head_woke == p)
      head_woke = p->next_woke;
    p->prev_woke = NULL;
    p->next_woke = NULL;
  }

  void link_proc(Process *p) {
    p->prev_proc = NULL;
    p->next_proc = head_proc;
    if (head_proc) {
      assert(!head_proc->prev_proc);
      head_proc->prev_proc = p;
    }
    head_proc = p;
  }

  void unlink_proc(Process *p) {
    if (p->next_proc)
      p->next_proc->prev_proc = p->prev_proc;
    if (p->prev_proc)
      p->prev_proc->next_proc = p->next_proc;
    if (head_proc == p)
      head_proc = p->next_proc;
    p->prev_proc = NULL;
    p->next_proc = NULL;
  }
};

}

#endif

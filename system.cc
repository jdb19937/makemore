#define __MAKEMORE_SYSTEM_CC__ 1

#include <stdlib.h>

#include "system.hh"
#include "process.hh"

namespace makemore {

using namespace std;

System::System() {
  head_woke = NULL;
  head_done = NULL;
  head_proc = NULL;
}

System::~System() {
  while (head_proc)
    delete head_proc;

  assert(!head_woke);
  assert(!head_proc);
  assert(!head_done);
}

void System::run() {
  for (Process *p = head_woke, *nextp; p; p = nextp) {
    assert(p->system == this);
    assert(p->woke);
    nextp = p->next_woke;
    fprintf(stderr, "running process [%s]\n", joinwords(p->args).c_str());
    p->run();
  }

  while (Process *p = head_done) {
    assert(p->system == this);
    fprintf(stderr, "deleting process [%s]\n", joinwords(p->args).c_str());
    assert(p->mode == Process::MODE_DONE);
    delete p;
  }
}

}

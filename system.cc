#define __MAKEMORE_SYSTEM_CC__ 1

#include <stdlib.h>

#include "system.hh"
#include "process.hh"
#include "tmutils.hh"

namespace makemore {

using namespace std;

System::System() {
  server = NULL;
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
  const unsigned int iters = 16;

  for (unsigned int i = 0; (head_woke || head_done || schedule.begin() != schedule.end()) && i < iters; ++i) {
    for (Process *nextp, *p = head_woke; p; p = nextp) {
      assert(p->system == this);
      assert(p->woke);
      nextp = p->next_woke;
      // fprintf(stderr, "running process [%s]\n", joinwords(p->args).c_str());
      p->run();
    }

    while (Process *p = head_done) {
      assert(p->system == this);
      // fprintf(stderr, "deleting process [%s]\n", joinwords(p->args).c_str());
      assert(p->mode == Process::MODE_DONE);
      delete p;
    }

    double t = now();
    while (schedule.begin() != schedule.end()) {
      auto s = schedule.begin();

      double when = s->first;
fprintf(stderr, "when=%lf t=%lf\n", when, t);
      if (when > t)
        break;
      Process *p = s->second;
      assert(p->mode == Process::MODE_WAITING);
fprintf(stderr, "running when=%lf t=%lf\n", when, t);
      p->run();

      schedule.erase(s);
    }
  }
}

}

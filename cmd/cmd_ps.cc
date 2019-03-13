#include <system.hh>
#include <urbite.hh>
#include <process.hh>
#include <strutils.hh>

using namespace makemore;

extern "C" void mainmore(Process *);

void mainmore(
  Process *process
) {
  System *system = process->system;
  Process *head_proc = system->head_proc;

  unsigned int i = 0;
  for (Process *proc = head_proc; proc; proc = proc->next) {
    char buf[256];

    strvec psvec;
    psvec.resize(2);
    
    psvec[0] = 

    (void)process->write(psvec);
  }
}

#include <system.hh>
#include <urbite.hh>
#include <process.hh>
#include <strutils.hh>
#include <session.hh>

using namespace makemore;

extern "C" void mainmore(Process *);

void mainmore(
  Process *process
) {
  Process *head_sproc = process->session->head_sproc;

  for (Process *proc = head_sproc; proc; proc = proc->next_sproc) {
    char buf[256];

    strvec psvec;
    psvec.resize(3);
    
    sprintf(buf, "0x%016lx", (uint64_t)proc);
    psvec[0] = buf;
    psvec[1] = proc->session->who->nom;
    psvec[2] = proc->cmd;
    catstrvec(psvec, proc->args);

    if (!process->write(psvec))
      break;
  }
}

#include <system.hh>
#include <urbite.hh>
#include <process.hh>
#include <strutils.hh>

using namespace makemore;

extern "C" void mainmore(Process *);

void mainmore(
  Process *process
) {
  while (strvec *inp = process->read()) {
    if (!process->write(*inp))
      break;
  }

  if (process->inproc) {
    process->inproc->outproc = NULL;
    process->inproc = NULL;
  }
  assert(!process->inagent);

  Command shellmore = find_command("sh");
  shellmore(process);
  assert(0);
}

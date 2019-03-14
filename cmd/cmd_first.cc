#include <system.hh>
#include <urbite.hh>
#include <process.hh>
#include <strutils.hh>

using namespace makemore;

extern "C" void mainmore(Process *);

void mainmore(
  Process *process
) {
  Command shellmore = find_command("sh");
  process->cmd = "sh";
  process->func = shellmore;
  shellmore(process);
  process->cmd = "first";
  process->func = mainmore;

fprintf(stderr, "first here\n");

  while (strvec *inp = process->read()) {
fprintf(stderr, "got line\n");
    if (!process->write(*inp))
      break;
  }
fprintf(stderr, "first there\n");
}

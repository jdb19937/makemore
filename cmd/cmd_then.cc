#include <system.hh>
#include <urbite.hh>
#include <process.hh>
#include <strutils.hh>

using namespace makemore;

extern "C" void mainmore(Process *);

void mainmore(
  Process *process
) {

fprintf(stderr, "in then\n");
  while (strvec *inp = process->peek()) {
fprintf(stderr, "then got line\n");
    if (!process->write(*inp))
      break;

    assert(process->read());
  }
fprintf(stderr, "out then\n");

  Command shellmore = find_command("sh");
  process->cmd = "sh";
  process->func = shellmore;
  shellmore(process);
  process->cmd = "then";
  process->func = mainmore;
}

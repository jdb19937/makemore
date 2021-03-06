#include <system.hh>
#include <urbite.hh>
#include <process.hh>
#include <strutils.hh>

using namespace makemore;

extern "C" void mainmore(Process *);

void mainmore(
  Process *process
) {
  unsigned int itabn = process->itab.size();
  if (itabn < 2)
    return;

  while (Line *inw = process->read(itabn - 1)) {
fprintf(stderr, "then got line\n");
    if (!process->write(inw)) {
      delete inw;
      break;
    }
  }
fprintf(stderr, "out then\n");

  process->itab[itabn - 1]->unlink_reader(process);
  process->itab.resize(itabn - 1);
fprintf(stderr, "out then2\n");

  Command shellmore = find_command("sh");
  process->cmd = "sh";
  process->func = shellmore;

  shellmore(process);

  process->cmd = "then";
  process->func = mainmore;
}

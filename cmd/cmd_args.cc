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

  strvec argext;
  if (!process->read(&argext, itabn - 1))
    return;

  process->itab[itabn - 1]->unlink_reader(process);
  process->itab.resize(itabn - 1);

  Command shellmore = find_command("sh");
  process->cmd = "sh";
  process->func = shellmore;
  strvec bak_args = process->args;
  catstrvec(process->args, argext);

  shellmore(process);

  process->cmd = "args";
  process->func = mainmore;
  process->args = bak_args;
}

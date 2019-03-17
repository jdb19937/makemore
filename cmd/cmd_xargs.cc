#include <system.hh>
#include <urbite.hh>
#include <process.hh>
#include <strutils.hh>

using namespace makemore;
using namespace std;

extern "C" void mainmore(Process *);

void mainmore(
  Process *process
) {
  unsigned int itabn = process->itab.size();
  process->itab.resize(itabn + 1);
  process->itab[itabn] = process->itab[0];
  process->itab[0] = new IO;
  process->itab[0]->link_reader(process);

  strvec bak_args = process->args;
  string bak_cmd = process->cmd;
  Command bak_func = process->func;

  Command shellmore = find_command("sh");
  process->cmd = "sh";
  process->func = shellmore;

  strvec invec;
  while (process->read(&invec, itabn)) {
    process->args = bak_args;
    catstrvec(process->args, invec);

    shellmore(process);
  }

  process->itab[0]->unlink_reader(process);
  process->itab[0] = process->itab[itabn];
  process->itab.resize(itabn);

  process->cmd = bak_cmd;
  process->func = bak_func;
  process->args = bak_args;
}

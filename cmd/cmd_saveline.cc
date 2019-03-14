#include <system.hh>
#include <urbite.hh>
#include <process.hh>
#include <strutils.hh>
#include <session.hh>

using namespace makemore;
using namespace std;

extern "C" void mainmore(Process *);

void mainmore(
  Process *process
) {
  if (process->args.size() != 1)
    return;

  if (!process->session->save_line(process->args[0]))
    return;

  strvec outvec;
  outvec.resize(1);
  outvec[0] = string("@") + process->args[0];
  (void) process->write(outvec);
}

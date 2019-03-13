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
  if (process->args.size() != 2)
    return;

  process->session->wordvar[ process->args[0] ] = process->args[1];

  strvec outvec;
  outvec.resize(1);
  outvec[0] = string("$") + process->args[0];
  process->write(outvec);
}

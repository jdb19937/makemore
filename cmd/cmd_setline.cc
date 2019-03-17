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
  if (process->args.size() < 1)
    return;

  strvec val(process->args.begin() + 1, process->args.end());

  Line lineval;
  strvec_to_line(val, &lineval);
  process->session->linevar[ process->args[0] ] = lineval;

  strvec outvec;
  outvec.resize(1);
  outvec[0] = string("@") + process->args[0];
  process->write(outvec);
}

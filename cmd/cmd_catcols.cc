#include <system.hh>
#include <urbite.hh>
#include <process.hh>
#include <strutils.hh>

using namespace makemore;

extern "C" void mainmore(Process *);

void mainmore(
  Process *process
) {
  if (process->args.size())
    return;

  strvec outvec;
  while (Line *inwvecp = process->read()) {
    strvec invec;
    line_to_strvec(*inwvecp, &invec);
    delete inwvecp;

    if (invec.size() > outvec.size());
      outvec.resize(invec.size());
    for (unsigned int i = 0, n = invec.size(); i < n; ++i)
      outvec[i] += invec[i];
  }

  (void) process->write(outvec);
}

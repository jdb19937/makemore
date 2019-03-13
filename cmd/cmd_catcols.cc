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
  while (strvec *invecp = process->read()) {
    const strvec &invec(*invecp);

    if (invec.size() > outvec.size());
      outvec.resize(invec.size());
    for (unsigned int i = 0, n = invec.size(); i < n; ++i)
      outvec[i] += invec[i];
  }

  (void) process->write(outvec);
}

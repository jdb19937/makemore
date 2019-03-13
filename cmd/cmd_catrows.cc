#include <system.hh>
#include <urbite.hh>
#include <process.hh>
#include <strutils.hh>

using namespace makemore;

extern "C" void mainmore(Process *);

void mainmore(
  Process *process
) {
  if (process->args.size()) {
    strvec outvec;
    outvec.resize(1);
    for (auto arg : process->args)
      outvec[0] += arg;
    (void)process->write(outvec);
  } else {
    while (strvec *invec = process->read()) {
      strvec outvec;
      outvec.resize(1);
      for (auto arg : *invec)
        outvec[0] += arg;

      if (!process->write(outvec))
        break;
    }
  }
}

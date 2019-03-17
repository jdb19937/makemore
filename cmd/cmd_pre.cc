#include <system.hh>
#include <urbite.hh>
#include <process.hh>
#include <strutils.hh>

#include <regex>

using namespace makemore;
using namespace std;

extern "C" void mainmore(
  Process *process
);

void mainmore(
  Process *process
) {
  strvec invec;
  while (process->read(&invec)) {
    strvec outvec = process->args;
    catstrvec(outvec, invec);

    if (!process->write(outvec))
      break;
  }
}


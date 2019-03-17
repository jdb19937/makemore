#include <system.hh>
#include <urbite.hh>
#include <process.hh>
#include <strutils.hh>

using namespace makemore;
using namespace std;

extern "C" void mainmore(
  Process *process
);

void mainmore(
  Process *process
) {
  
  strvec invec;
  if (!process->read(&invec))
    return;

  strvec prev = invec;
  if (!process->write(prev))
    return;

  while (process->read(&invec)) {
    if (invec == prev)
      continue;

    prev = invec;
    if (!process->write(prev))
      break;
  }
}


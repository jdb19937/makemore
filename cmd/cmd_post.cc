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
  strvec in;
  while (process->read(&in)) {
    catstrvec(in, process->args);

    if (!process->write(in))
      break;
  }
}


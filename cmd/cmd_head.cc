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
  if (process->args.size() == 0)
    return;

  unsigned long nrows = strtoul(process->args[0].c_str(), NULL, 0);
  unsigned long row = 0;

  while (row < nrows) {
    Line *w;
    if (!(w = process->read()))
      break;
    if (!process->write(w)) {
      delete w;
      break;
    }

    ++row;
  }
}


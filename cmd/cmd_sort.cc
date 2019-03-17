#include <system.hh>
#include <urbite.hh>
#include <process.hh>
#include <strutils.hh>

#include <list>
#include <algorithm>
#include <vector>
#include <string>

using namespace makemore;
using namespace std;

extern "C" void mainmore(
  Process *process
);

void mainmore(
  Process *process
) {
  std::list<strvec> sortbuf;

  strvec invec;
  while (process->read(&invec)) {
    sortbuf.push_back(invec);
  }

  sortbuf.sort();

  for (auto outvec : sortbuf) {
    if (!process->write(outvec))
      break;
  }
}


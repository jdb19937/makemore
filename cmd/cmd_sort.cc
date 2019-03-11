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

  while (strvec *inp = process->read()) {
    sortbuf.push_back(*inp);
  }

  sortbuf.sort();

  for (auto outvec : sortbuf) {
    if (!process->write(outvec))
      break;
  }

  process->coro->finish();
}


#include <system.hh>
#include <urbite.hh>
#include <process.hh>
#include <strutils.hh>
#include <session.hh>
#include <cudamem.hh>

using namespace makemore;
using namespace std;

extern "C" void mainmore(
  Process *process
);

void mainmore(
  Process *process
) {
  while (Line *line = process->read()) {
    for (Word &word : *line) {
      word.cudify();
    }

    if (!process->write(line)) {
      delete line;
      break;
    }
  }
}


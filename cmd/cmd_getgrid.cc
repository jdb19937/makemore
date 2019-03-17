#include <system.hh>
#include <urbite.hh>
#include <process.hh>
#include <strutils.hh>
#include <session.hh>

using namespace makemore;
using namespace std;

extern "C" void mainmore(Process *);

void mainmore(
  Process *process
) {
  if (process->args.size() != 1)
    return;

  const Grid &grid = process->session->gridvar[ process->args[0] ];

  for (auto line : grid) {
    Line *nline = new Line(line);
    if (!process->write(nline)) {
      delete nline;
      break;
    }
  }
}

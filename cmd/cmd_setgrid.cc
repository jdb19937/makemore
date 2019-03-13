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

  strmat &grid = process->session->gridvar[ process->args[0] ];

  grid.clear();
  while (strvec *invec = process->read()) {
    grid.push_back(*invec);
  }

  strvec outvec;
  outvec.resize(1);
  outvec[0] = string("%") + process->args[0];
  process->write(outvec);
}

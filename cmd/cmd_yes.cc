#include <system.hh>
#include <urbite.hh>
#include <process.hh>
#include <strutils.hh>

using namespace makemore;

extern "C" void mainmore(Process *);

void mainmore(
  Process *process
) {
  strvec yes;
  if (process->args.size()) {
    yes = process->args;
  } else {
    yes.resize(1);
    yes[0] = "y";
  }

  while (process->write(yes))
    ;

fprintf(stderr, "finishing yes\n");
}

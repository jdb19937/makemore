#include <system.hh>
#include <urbite.hh>
#include <process.hh>
#include <strutils.hh>

using namespace makemore;

extern "C" void mainmore(Process *);

void mainmore(
  Process *process
) {
fprintf(stderr, "here echo args=%s\n", joinwords(process->args).c_str());

  (void)process->write(process->args);

fprintf(stderr, "finishing echo\n");
}

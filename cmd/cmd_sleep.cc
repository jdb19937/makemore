#include <system.hh>
#include <urbite.hh>
#include <process.hh>
#include <strutils.hh>
#include <tmutils.hh>

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

  double dt = strtod(process->args[0].c_str(), NULL);

fprintf(stderr, "sleeping %lf\n", dt);

  process->sleep(dt);
fprintf(stderr, "done with sleep\n");
}


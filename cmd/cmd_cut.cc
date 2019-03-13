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

  string colspec = process->args[0];
  unsigned int col = strtoul(colspec.c_str(), NULL, 0);

  while (strvec *inp = process->read()) {
fprintf(stderr, "cut got inp\n");
    strvec outvec;
    outvec.resize(1);
    outvec[0] = col < inp->size() ? (*inp)[col] : "";

    if (!process->write(outvec))
      break;
  }
fprintf(stderr, "cut there\n");

  return;
}


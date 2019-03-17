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
  vector<int> cols;
  parsecolspec(colspec, &cols);
  unsigned int ncols = cols.size();

  strvec invec;
  while (process->read(&invec)) {
    unsigned int kcols = invec.size();

    strvec outvec;
    outvec.resize(ncols);
    for (unsigned int i = 0; i < ncols; ++i) {
      unsigned int col = cols[i];
      outvec[i] = col < kcols ? invec[col] : "";
    }

    if (!process->write(outvec))
      break;
  }

  return;
}


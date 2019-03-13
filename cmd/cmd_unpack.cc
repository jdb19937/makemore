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
  char buf[64];

  while (const strvec *invecp = process->read()) {
    const strvec &invec = *invecp;

    if (invec.size() == 0) {
      continue;
    }

    const string &instr = invec[0];
    unsigned int outvecn = invec[0].length() / sizeof(double);
    const double *inpack = (double *)instr.data();

    strvec outvec;
    outvec.resize(outvecn);

    for (unsigned int i = 0; i < outvecn; ++i) {
      sprintf(buf, "%.17g", inpack[i]);
      outvec[i] = string(buf);
    }

    if (!process->write(outvec))
      break;
  }

  return;
}


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
  while (const strvec *invecp = process->read()) {
    const strvec &invec = *invecp;
    unsigned int invecn = invec.size();

    strvec outvec;
    outvec.resize(1);
    string &outstr = outvec[0];
    outstr.resize(invecn * sizeof(double));
    double *outpack = (double *)outstr.data();

    for (unsigned int i = 0; i < invecn; ++i) {
      outpack[i] = strtod(invec[i].c_str(), NULL);
    }

    if (!process->write(outvec))
      break;
  }

  return;
}


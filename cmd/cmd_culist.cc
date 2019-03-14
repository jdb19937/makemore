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
  char buf[64];

  strvec outvec;
  outvec.resize(1);
  for (auto ptrlen : process->session->cudavar) {
    sprintf(buf, "*%016lx+%u", (uint64_t)ptrlen.first, (unsigned int)ptrlen.second);
    outvec[0] = buf;
    if (!process->write(outvec))
      break;
  }
}


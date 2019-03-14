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

  while (const strvec *invecp = process->read()) {
    const strvec &invec = *invecp;

    if (invec.size() == 0)
      continue;

    const string &instr = invec[0];
    const uint8_t *instrd = (const uint8_t *)instr.data();
    unsigned int instrn = instr.length();

    uint8_t *cuvar = (uint8_t *)process->session->cumakevar(instrn);
    makemore::encude(instrd, instrn, cuvar);

    char cuvarbuf[64];
    sprintf(cuvarbuf, "*%016lx+%u", (uint64_t)cuvar, instrn);

    strvec outvec;
    outvec.resize(1);
    outvec[0] = cuvarbuf;

    if (!process->write(outvec))
      break;
  }

  return;
}


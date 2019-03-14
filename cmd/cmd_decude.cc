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

    std::string cuvarstr = invec[0];
    if (cuvarstr.length() == 0 || cuvarstr[0] != '*')
      continue;
    const char *cuvarlenp = strchr(cuvarstr.c_str() + 1, '+');
    if (!cuvarlenp)
      continue;
    string cuvarstrpos(cuvarstr.c_str() + 1, cuvarlenp - cuvarstr.c_str() - 1);
    uint64_t cuvarpos = strtoul(cuvarstrpos.c_str(), NULL, 16);
    unsigned int cuvarlen = strtoul(cuvarlenp + 1, NULL, 0);

    strvec outvec;
    outvec.resize(1);
    string &outstr = outvec[0];

    outstr.resize(cuvarlen);
fprintf(stderr, "cuvarpos=%016lx cuvarlen=%u\n", cuvarpos, cuvarlen);
    makemore::decude((uint8_t *)cuvarpos, cuvarlen, (uint8_t *)outstr.data());
    
    if (!process->write(outvec))
      break;
  }

  return;
}


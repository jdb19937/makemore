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

  for (auto ptrlen : process->session->cudavar) {
    uint64_t cuvarpos = (uint64_t)ptrlen.first;
    unsigned int cuvarlen = ptrlen.second;

    cufree((void *)cuvarpos);
  }

  process->session->cudavar.clear();
}


#include <system.hh>
#include <urbite.hh>
#include <process.hh>
#include <strutils.hh>

using namespace makemore;

extern "C" void mainmore(Process *);

void mainmore(
  Process *process
) {
  strvec out;
  out.resize(1);

  unsigned int i = 0;
  char buf[64];

  while (strvec *inp = process->read()) {
    ++i;
  }

  sprintf(buf, "%u", i);
  out[0] = buf;

  (void) process->write(out);
}

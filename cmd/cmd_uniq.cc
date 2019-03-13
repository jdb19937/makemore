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
  strvec *inp = process->read();
  if (!inp)
    return;

  strvec prev = *inp;
  if (!process->write(prev))
    return;

  while (inp = process->read()) {
    const strvec &in = *inp;
fprintf(stderr, "here [%s]\n", joinwords(in).c_str());
    if (in == prev)
      continue;

    prev = in;
    if (!process->write(prev))
      break;
  }
}


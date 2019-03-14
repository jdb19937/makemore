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

  if (process->args.size() == 0) {
    while (strvec *inp = process->read()) {
      sprintf(buf, "%u", i);
      out[0] = buf;
      catstrvec(out, *inp);

      if (!process->write(out))
        break;
      ++i;
    }
  } else {
    unsigned int n = (unsigned int)strtoul(process->args[0].c_str(), NULL, 0);
    for (i = 0; i < n; ++i) {
      sprintf(buf, "%u", i);
      out[0] = buf;
      if (!process->write(out))
        break;
    }
  }
}

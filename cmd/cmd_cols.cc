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

  unsigned int outbufn = strtoul(process->args[0].c_str(), NULL, 0);
  if (outbufn == 0)
    return;

  strvec outbuf;
  outbuf.resize(outbufn);
  unsigned int outbufi = 0;
  
  while (Line *inwvecp = process->read()) {
    strvec invec;
    line_to_strvec(*inwvecp, &invec);
    delete inwvecp;

    for (auto ini = invec.begin(); ini != invec.end(); ++ini) {
      if (outbufi == outbufn) {
        if (!process->write(outbuf))
          goto done;
        outbufi = 0;
      }

      outbuf[outbufi++] = *ini;
    }
  }

  if (outbufi) {
    outbuf.resize(outbufi);
    process->write(outbuf);
  }

done:
  return;
}


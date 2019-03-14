#include <system.hh>
#include <server.hh>
#include <system.hh>
#include <urbite.hh>
#include <process.hh>
#include <strutils.hh>

using namespace makemore;

extern "C" void mainmore(Process *);

void mainmore(
  Process *process
) {
  if (process->args.size() == 0)
    return;
  unsigned int n = strtoul(process->args[0].c_str(), NULL, 0);

  Urb *urb = process->system->server->urb;
  assert(urb);
  urb->restock(n);

  strvec outvec;
  outvec.resize(1);
  outvec[0] = "ok";
  (void) process->write(outvec);
}

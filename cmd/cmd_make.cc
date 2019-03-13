#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include <string>
#include <set>
#include <vector>

#include <system.hh>
#include <urbite.hh>
#include <process.hh>
#include <strutils.hh>
#include <strutils.hh>
#include <parson.hh>
#include <org.hh>
#include <zone.hh>
#include <server.hh>
#include <ppm.hh>
#include <imgutils.hh>

namespace makemore {

using namespace makemore;
using namespace std;

extern "C" void mainmore(Process *);

void mainmore(
  Process *process
) {
  const strvec &arg = process->args;

  if (arg.size() != 0 && arg.size() != 1) {
    strvec errvec;
    splitwords("bad args", &errvec);
    process->write(errvec);
    return;
  }

  Server *server = process->system->server;
  assert(server);
  Urb *urb = server->urb;
  assert(urb);

  std::string newnom;
  if (arg.size() == 1)
    newnom = arg[0];
  else 
    newnom = Parson::gen_nom();

  if (urb->find(newnom)) {
    strvec errvec;
    splitwords("already exists", &errvec);
    process->write(errvec);
    return;
  }

  if (!Parson::valid_nom(newnom.c_str())) {
    strvec errvec;
    splitwords("bad nom", &errvec);
    process->write(errvec);
    return;
  }

#if 0
  Parson *nx = urb->make(newnom);
  nx->created = time(NULL);
  nx->creator = process->outagent ? process->outagent->ip : 0x7F000001;
  nx->revised = time(NULL);
  nx->revisor = process->outagent ? process->outagent->ip : 0x7F000001;
#endif
  urb->make(newnom);

  {
    strvec outvec;
    outvec.resize(2);
    outvec[0] = "made";
    outvec[1] = newnom;
    process->write(outvec);
  }
}

}

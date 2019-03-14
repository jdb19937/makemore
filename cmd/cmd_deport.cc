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
#include <session.hh>
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
  unsigned int zid = 0;
  strvec &arg = process->args;
  if (arg.size() != 1 && arg.size() != 2 && arg.size() != 3) {
    strvec errvec;
    splitwords("error bad args", &errvec);
    process->write(errvec);
    return;
  }

  Server *server = process->system->server;
  assert(server);
  Urb *urb = server->urb;
  assert(urb);

  std::string nom = arg[0];
  Parson *parson = urb->find(nom);
  if (!parson) {
    strvec errvec;
    splitwords("error bad nom", &errvec);
    process->write(errvec);
    return;
  }

  urb->deport(nom.c_str());

  strvec outvec;
  outvec.resize(1);
  outvec[0] = "ok";
  process->write(outvec);
}

}

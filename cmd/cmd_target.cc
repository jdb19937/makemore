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

  if (arg.size() != 1)
    return;

  Server *server = process->system->server;
  assert(server);
  Urb *urb = server->urb;
  assert(urb);

  string nom = arg[0];

  Parson *parson = urb->find(nom);
  if (!parson)
    return;

  string png;
  //labpng(parson->target, Parson::dim, Parson::dim, &png);

  strvec outvec;
  outvec.resize(1);
  outvec[0] = png;
  (void) process->write(outvec);
}

}

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
  unsigned int npicks = 1;
  if (process->args.size() > 0)
    npicks = (unsigned int)strtoul(process->args[0].c_str(), NULL, 0);

  unsigned int zid = 0;
  if (process->args.size() > 1)
    zid = (unsigned int)strtoul(process->args[1].c_str(), NULL, 0);

  Server *server = process->system->server;
  assert(server);
  Urb *urb = server->urb;
  assert(urb);

  if (zid >= urb->zones.size()) {
    return;
  }

  Zone *zone = urb->zones[zid];
  assert(zone);

  for (unsigned int i = 0; i < npicks; ++i) {
    Parson *pick;
    pick = zone->pick();
    if (!pick)
      break;

    strvec outvec;
    outvec.resize(1);
    outvec[0] = string(pick->nom);

    if (!process->write(outvec))
      break;
  }
}

}

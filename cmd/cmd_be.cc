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

  std::string newnom = arg[0];
  Parson *parson = urb->find(newnom);
  if (!parson) {
    strvec errvec;
    splitwords("error bad nom", &errvec);
    process->write(errvec);
    return;
  }

  string sespass;
  if (arg.size() == 2)
    sespass = arg[1];

  string session;
  if (parson->has_pass()) {
    if (sespass.length() == 80) {
      session = sespass;
      if (!server->check_session(newnom, session)) {
        strvec errvec;
        splitwords("error bad session", &errvec);
        process->write(errvec);
        return;
      }
    } else {
      string pass = sespass;
      if (!parson->check_pass(pass)) {
        strvec errvec;
        splitwords("error bad pass", &errvec);
        process->write(errvec);
        return;
      }

      unsigned long duration = 86400;
      if (arg.size() == 3)
        duration = strtoul(arg[2].c_str(), NULL, 0);

      session = server->make_session(newnom, duration);
    }
  } else {
    session = sespass;
    if (!server->check_session(newnom, session)) {
      unsigned long duration = 1UL << 48;
      if (arg.size() == 3)
        duration = strtoul(arg[2].c_str(), NULL, 0);

      session = server->make_session(newnom, duration);
    }
  }

  strvec outvec;
  outvec.resize(2);
  outvec[0] = "session";
  outvec[1] = session;
  process->write(outvec);

  process->system->server->renom(process->session->agent, newnom);

  return;
}

}

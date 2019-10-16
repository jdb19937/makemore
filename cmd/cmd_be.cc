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
#include <zone.hh>
#include <server.hh>
#include <ppm.hh>
#include <imgutils.hh>
#include <agent.hh>

namespace makemore {

using namespace makemore;
using namespace std;

extern "C" void mainmore(Process *);

void mainmore(
  Process *process
) {
  unsigned int zid = 0;
  strvec &arg = process->args;

if (arg.size() > 0)
fprintf(stderr, "arg0=%lu\n", arg[0].length());
if (arg.size() > 1)
fprintf(stderr, "arg1=%lu\n", arg[1].length());
if (arg.size() > 2)
fprintf(stderr, "arg2=%lu\n", arg[2].length());

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
  if (!Parson::valid_nom(newnom)) {
    strvec errvec;
    splitwords("error bad nom", &errvec);
    process->write(errvec);
    return;
  }
  Parson *parson = urb->make(newnom, 0);
  if (!parson) {
    strvec errvec;
    splitwords("error bad nom", &errvec);
    process->write(errvec);
    return;
  }

  std::string newpubkey;
  if (arg.size() == 3) {
    newpubkey = arg[2];
fprintf(stderr, "newpubkeylen=%lu\n", newpubkey.length());
    if (newpubkey.length() != 128) {
      strvec errvec;
      splitwords("error bad pubkey", &errvec);
      process->write(errvec);
      return;
    }
  }

  string sespass;
  if (arg.size() >= 2)
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

      if (!*parson->owner)
        strcpy(parson->owner, parson->nom);

      unsigned long duration = 86400;
      session = server->make_session(newnom, duration);
    }
  } else {
    session = sespass;
    if (!server->check_session(newnom, session)) {
      if (newpubkey.length() != 128) {
        strvec errvec;
        splitwords("error bad pubkey2", &errvec);
        process->write(errvec);
        return;
      }

      std::string newpass = sespass;
      parson->set_pass(newpass);
      memcpy(parson->pubkey, newpubkey.data(), newpubkey.length());
      strcpy(parson->owner, parson->nom);
      
      unsigned long duration = 1UL << 48;
      session = server->make_session(newnom, duration);
    }
  }

  parson->acted = time(NULL);

  strvec outvec;
  outvec.resize(2);
  outvec[0] = "session";
  outvec[1] = session;
  process->write(outvec);

  Agent *agent = process->session->agent;
  server->renom(agent, newnom);

  // process->session->loadvars();

  return;
}

}

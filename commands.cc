#define __MAKEMORE_COMMANDS_CC__ 1
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include <string>
#include <vector>

#include "commands.hh"
#include "strutils.hh"
#include "parson.hh"
#include "org.hh"
#include "zone.hh"
#include "server.hh"
#include "ppm.hh"

namespace makemore {

int _startup_count = 0;

using namespace std;

NEW_CMD(be) {
  if (args.size() != 1)
    return;

  std::string newnom = args[0];
  Parson *parson = agent->server->urb->find(newnom);
  if (!parson) {
    agent->write("not here\n");
    return;
  }

  agent->server->renom(agent, newnom);
  agent->write("ok\n");
}

NEW_CMD(make) {
  if (args.size() != 1)
    return;

  std::string newnom = args[0];
  if (agent->server->urb->find(newnom)) {
    agent->write("already here\n");
    return;
  }

  if (!Parson::valid_nom(newnom.c_str())) {
    agent->write("bad nom\n");
    return;
  }

  Parson x(newnom.c_str());
  Parson *nx = agent->server->urb->make(x);
  nx->created = time(NULL);
  nx->creator = agent->ip;
  nx->revised = time(NULL);
  nx->revisor = agent->ip;


  agent->write("ok\n");
}


NEW_CMD(exit) {
  ::close(agent->s);
}
  
NEW_CMD(echo) {
  if (args.size() == 0)
    return;

  std::string out = args[0];
  for (unsigned int i = 1, n = args.size(); i < n; ++i) {
    out += " ";
    out += args[i];
  }
  out += "\n";

  agent->write(out);
}

NEW_CMD(burn) {
  Urb *urb = agent->server->urb;

  if (args.size() < 1)
    return;
  unsigned int n = (unsigned int)atoi(args[0].c_str());
  if (n > 65535)
    n = 65535;

  double nu = 0.0001;
  double pi = 0.0001;
  if (args.size() >= 2)
    nu = strtod(args[1].c_str(), NULL);
  if (args.size() >= 3)
    pi = strtod(args[2].c_str(), NULL);

  std::vector<Parson*> parsons;
  parsons.resize(n);
  Zone *zone = urb->zones[0];
  for (unsigned int i = 0; i < n; ++i)
    parsons[i] = zone->pick();

  urb->pipex->burn(parsons.data(), parsons.size(), nu, pi);
  urb->pipex->save();
  // urb->pipex->report("burn", outfp);
}

NEW_CMD(to) {
  Server *server = agent->server;
  Urb *urb = server->urb;

  if (args.size() < 2) {
    agent->printf("who\n");
    return;
  }

  string nom = args[0];
  string cmd_from_nom = string("from ") + nom + string(" ");
  string cmd_to_nom = string("to ") + nom + string(" ");

  string msg = "";
  for (unsigned int i = 0; i < thread.size(); ++i) {
    string oldmsg = thread[i];
    if (!strncmp(oldmsg.c_str(), cmd_from_nom.c_str(), cmd_from_nom.length())) {
      msg += string(msg.length() ? ", " : "") + 
        string("to ") + agent->who->nom + string(" ") +
        string(oldmsg.c_str() + cmd_from_nom.length());
    } if (!strncmp(oldmsg.c_str(), cmd_to_nom.c_str(), cmd_to_nom.length())) {
      msg += string(msg.length() ? ", " : "") + 
        string("from ") + agent->who->nom + string(" ") +
        string(oldmsg.c_str() + cmd_to_nom.length());
    }
  }

  msg += string(msg.length() ? ", " : "") + string("from ") + agent->who->nom;
  for (unsigned int i = 1, n = args.size(); i < n; ++i)
    msg += string(" ") + args[i];

  if (Parson *to = urb->find(nom)) {
    to->pushbrief(msg);
    agent->printf("sent\n");
  } else {
    agent->printf("lost\n");
  }
}

}

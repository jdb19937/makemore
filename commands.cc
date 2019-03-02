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
  if (arg.size() != 1)
    return;

  std::string newnom = arg[0];
  Parson *parson = agent->server->urb->find(newnom);
  if (!parson) {
    agent->write("not here\n");
    return;
  }

  agent->server->renom(agent, newnom);
  agent->write("ok\n");
}

NEW_CMD(make) {
  if (arg.size() != 1)
    return;

  std::string newnom = arg[0];
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
  if (arg.size() == 0)
    return;

  std::string out = arg[0];
  for (unsigned int i = 1, n = arg.size(); i < n; ++i) {
    out += " ";
    out += arg[i];
  }
  out += "\n";

  agent->write(out);
}

NEW_CMD(burn) {
  Urb *urb = agent->server->urb;

  if (arg.size() < 1)
    return;
  unsigned int n = (unsigned int)atoi(arg[0].c_str());
  if (n > 65535)
    n = 65535;

  double nu = 0.0001;
  double pi = 0.0001;
  if (arg.size() >= 2)
    nu = strtod(arg[1].c_str(), NULL);
  if (arg.size() >= 3)
    pi = strtod(arg[2].c_str(), NULL);

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

  if (arg.size() < 2) {
    agent->printf("who\n");
    return;
  }

  string nom = arg[0];

  string msg = string("from ") + agent->who->nom;
  for (unsigned int i = 1, n = arg.size(); i < n; ++i)
    msg += string(" ") + arg[i];
  server->notify(nom, msg);

  for (unsigned int i = 0; i < ctx.size(); ++i) {
    const vector<string> &words = ctx[i];
    if (words.size() < 3)
      continue;

    if (words[0] == "from" && words[1] == nom) {
      vector<string> tailwords(words.begin() + 2, words.end());
      msg += string(" | to ") + agent->who->nom + " " + join(tailwords, " ");
    } if (words[0] == "to" && words[1] == nom) {
      vector<string> tailwords(words.begin() + 2, words.end());
      msg += string(" | from ") + agent->who->nom + " " + join(tailwords, " ");
    }
  }

  if (Parson *to = urb->find(nom)) {
    to->pushbrief(msg);
    agent->printf("sent\n");
  } else {
    agent->printf("lost\n");
  }
}

}

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
  if (arg.size() != 1 && arg.size() != 2)
    return;

  std::string newnom = arg[0];
  Parson *parson = agent->server->urb->find(newnom);
  if (!parson) {
    agent->write("not here\n");
    return;
  }

  string pass;
  if (arg.size() == 2)
    pass = arg[1];

  if (!parson->check_pass(pass)) {
    agent->write("bad pass\n");
    return;
  }

  agent->server->renom(agent, newnom);
  agent->write("ok\n");
}

NEW_CMD(target) {
  if (arg.size() != 2)
    return;
  const std::string &jpeg = arg[1];
  PPM ppm;

  ppm.read_jpeg(jpeg);
  if (ppm.w != Parson::dim || ppm.h != Parson::dim) {
    agent->printf("wrong dims\n");
    return;
  }

  const std::string &nom = arg[0];
  Parson *parson = agent->server->urb->find(nom);
  if (!parson) {
    agent->write("not here\n");
    return;
  }

  ppm.vectorize(parson->target);
  agent->write("ok\n");
}

NEW_CMD(tag) {
  if (arg.size() != 2)
    return;

  const std::string &nom = arg[0];
  Parson *parson = agent->server->urb->find(nom);
  if (!parson) {
    agent->write("bad nom\n");
    return;
  }

  const std::string &tag = arg[1];

  if (!Parson::valid_tag(tag.c_str())) {
    agent->write("bad tag\n");
    return;
  }

  parson->add_tag(tag.c_str());
  agent->write("ok\n");
}

NEW_CMD(tags) {
  if (arg.size() != 1)
    return;

  const std::string &nom = arg[0];
  if (!Parson::valid_nom(nom.c_str()))
    return;

  Parson *parson = agent->server->urb->find(nom);
  if (!parson) {
    agent->write("bad nom\n");
    return;
  }

  for (unsigned int i = 0; i < Parson::ntags; ++i) {
    if (*parson->tags[i]) {
      agent->printf("tag %s %s\n", nom.c_str(), parson->tags[i]);
    }
  }
}

NEW_CMD(pass) {
  if (arg.size() != 1 && arg.size() != 2)
    return;

  if (strchr(agent->who->nom.c_str(), '.')) {
    agent->printf("nope\n");
    return;
  }

  Parson *parson = agent->who->parson();
  if (!parson)
    return;

  string pass; 
  if (arg.size() == 2)
    pass = arg[1];
  parson->set_pass(pass);
  agent->printf("ok\n");
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

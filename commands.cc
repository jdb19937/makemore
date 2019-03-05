#define __MAKEMORE_COMMANDS_CC__ 1
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include <string>
#include <set>
#include <vector>

#include "commands.hh"
#include "strutils.hh"
#include "parson.hh"
#include "org.hh"
#include "zone.hh"
#include "server.hh"
#include "ppm.hh"
#include "imgutils.hh"

namespace makemore {

int _startup_count = 0;

using namespace std;

NEW_CMD(pick) {
  unsigned int npicks = 1;
  if (arg.size() > 0)
    npicks = (unsigned int)strtoul(arg[0].c_str(), NULL, 0);
  if (npicks > 256)
    npicks = 256;

  unsigned int zid = 0;
  if (arg.size() > 1)
    zid = (unsigned int)strtoul(arg[1].c_str(), NULL, 0);

  Urb *urb = agent->server->urb;
  assert(urb);

  if (zid >= urb->zones.size()) {
    agent->printf("bad zone\n");
    return;
  }

  Zone *zone = urb->zones[zid];
  assert(zone);

  vector<Urbite*> &pickbuf = agent->pickbuf;
  unsigned int tpicks = pickbuf.size();
  pickbuf.resize(tpicks + npicks);

  for (unsigned int i = 0; i < npicks; ++i) {
    Parson *pick = zone->pick();
    if (!pick)
      return;

    Urbite *upick = new Urbite(pick->nom, urb, pick);
    agent->printf("picked %s\n", upick->nom.c_str()); 
    pickbuf[tpicks + i] = upick;
  }
  tpicks += npicks;

  if (tpicks > Agent::maxpicks) {
    unsigned int remove = tpicks - Agent::maxpicks;
    for (unsigned int i = 0; i < Agent::maxpicks; ++i) {
      if (i < remove)
        delete pickbuf[i];
      pickbuf[i] = pickbuf[i + remove];
    }
    pickbuf.resize(Agent::maxpicks);
  }
}

NEW_CMD(http) {
  if (arg.size())
    return;

  vector<string> &httpbuf = agent->httpbuf;
  for (unsigned int i = 0, n = httpbuf.size(); i < n; ++i) {
    vector<string> words;
    splitwords(httpbuf[i], &words);
    unsigned int wordsn = words.size();
    if (wordsn == 0) {
      agent->close();
      return;
    }

    if (i == 0) {
      if (words[0] != "GET" || wordsn != 3) {
        agent->close();
        return;
      }

      agent->httpvers = words[2];
      if (agent->httpvers == "HTTP/1.0") {
        agent->httpkeep = false;
      } else if (agent->httpvers == "HTTP/1.1") {
        agent->httpkeep = true;
      } else {
        agent->close();
        return;
      }

      continue;
    }

    string key = lowercase(words[0]);
    unsigned int keyn = key.length();
    if (keyn == 0 || key[keyn - 1] != ':') {
      agent->close();
      return;
    }
    key.resize(keyn - 1);

    if (key == "connection") {
      if (words[1] == "keep-alive" && agent->httpvers == "HTTP/1.1") {
        agent->httpkeep = true;
      } else if (words[1] == "close") {
        agent->httpkeep = false;
      }
    } else if (key == "user-agent") {
      vector<string> rest;
      rest.resize(wordsn - 1);
      for (unsigned int i = 1; i < wordsn; ++i)
        rest[i - 1] = words[i];

      agent->httpua = joinwords(rest);
    }
  }

fprintf(stderr, "httpkeep=%d\nhttpua=%s\nhttpvers=%s\n", agent->httpkeep, agent->httpua.c_str(), agent->httpvers.c_str());
   

  std::string body = "hello there how are you?\n";

  agent->printf("%s 200 OK\r\n", agent->httpvers.c_str());
  agent->printf("Server: makemore\r\n");
  agent->printf("Content-Type: text/plain\r\n");
  agent->printf("Content-Length: %lu\r\n", body.length());
  if (agent->httpvers == "HTTP/1.1")
    agent->printf("Connection: %s\r\n", agent->httpkeep ? "keep-alive" : "close");
  agent->printf("\r\n");
  agent->write(body);

fprintf(stderr, "outbufn=%u httpbuf=[%s]\n", agent->outbufn, join(agent->httpbuf, "\n").c_str());
}

NEW_CMD(be) {
  if (arg.size() != 1 && arg.size() != 2 && arg.size() != 3) {
    agent->write("bad args\n");
    return;
  }

  std::string newnom = arg[0];
  Parson *parson = agent->server->urb->find(newnom);
  if (!parson) {
    agent->write("bad nom\n");
    return;
  }

  string sespass;
  if (arg.size() == 2)
    sespass = arg[1];

  string session;
  if (parson->has_pass()) {
    if (sespass.length() == 80) {
      session = sespass;
      if (!agent->server->check_session(newnom, session)) {
        agent->printf("bad session\n");
        return;
      }
    } else {
      string pass = sespass;
      if (!parson->check_pass(pass)) {
        agent->printf("bad pass\n");
        return;
      }

      unsigned long duration = 86400;
      if (arg.size() == 3)
        duration = strtoul(arg[2].c_str(), NULL, 0);

      session = agent->server->make_session(newnom, duration);
    }
  } else {
    session = sespass;
    if (!agent->server->check_session(newnom, session)) {
      unsigned long duration = 1UL << 48;
      if (arg.size() == 3)
        duration = strtoul(arg[2].c_str(), NULL, 0);

      session = agent->server->make_session(newnom, duration);
    }
  }

  agent->server->renom(agent, newnom);
  agent->printf("session %s\n", session.c_str());
}

NEW_CMD(target) {
  if (arg.size() != 2)
    return;
  const std::string &img = arg[1];

  const std::string &nom = arg[0];
  Parson *parson = agent->server->urb->find(nom);
  if (!parson) {
    agent->write("not here\n");
    return;
  }

  bool ret = imglab("png", img, Parson::dim, Parson::dim, parson->target);
  if (ret)
    agent->write("ok\n");
  else
    agent->write("not ok\n");
}

NEW_CMD(show) {
  if (arg.size() != 1)
    return;

  if (arg[0] == "target") {
    Parson *parson = agent->who->parson();

    string png;
    labimg(parson->target, Parson::dim, Parson::dim, "png", &png);

    char buf[64];
    sprintf(buf, "png <%lu\n", png.length());
    agent->write(buf);
    agent->write(png);

    return;
  }

  if (arg[0] == "partrait") {
    Parson *parson = agent->who->parson();
    agent->server->urb->generate(parson, 0);

    string png;
    labimg(parson->partrait, Parson::dim, Parson::dim, "png", &png);

    char buf[64];
    sprintf(buf, "png <%lu\n", png.length());
    agent->write(buf);
    agent->write(png);

    return;
  }
}

NEW_CMD(punch) {
  if (arg.size() != 1)
    return;

  const std::string &nom = arg[0];
  Parson *parson = agent->server->urb->find(nom);
  if (!parson) {
    agent->write("bad nom\n");
    return;
  }

  assert(Parson::dim >= 32);
  assert(Parson::dim % 4 == 0);
  unsigned int hdim = Parson::dim / 2;
  unsigned int qdim = hdim / 2;
  unsigned int x0 = -4 + qdim + (randuint() % hdim);
  unsigned int y0 = -4 + qdim + (randuint() % hdim);
  unsigned int x1 = x0 + 8;
  unsigned int y1 = y0 + 8;
  assert(x1 <= Parson::dim);
  assert(y1 <= Parson::dim);

  for (unsigned int y = y0; y < y1; ++y) {
    for (unsigned int x = x0; x < x1; ++x) {
      parson->target[y * Parson::dim * 3 + x * 3 + 0] = 0;
      parson->target[y * Parson::dim * 3 + x * 3 + 1] = 128;
      parson->target[y * Parson::dim * 3 + x * 3 + 2] = 128;
    }
  }


  agent->printf("bam\n");

  char buf[64];
  sprintf(buf, "punched by %s", agent->who->nom.c_str());
  string msg = buf;
  agent->server->notify(nom, msg);

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
  parson->pushbrief(msg);
}


NEW_CMD(untag) {
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

  parson->del_tag(tag.c_str());
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

NEW_CMD(cents) {
  if (arg.size())
    return;

  Parson *parson = agent->who->parson();
  if (!parson)
    return;

  agent->printf("%lu\n", parson->cents);
}

NEW_CMD(pay) {
  if (arg.size() != 2)
    return;

  Parson *from = agent->who->parson();
  if (!from)
    return;

  string nom = arg[0];
  Parson *to = agent->server->urb->find(nom);
  if (!to) {
    agent->printf("who\n");
    return;
  }

  uint64_t pay_amount = strtoull(arg[1].c_str(), NULL, 0);
  uint64_t from_amount = from->cents;
  uint64_t to_amount = to->cents;

  if (pay_amount > from_amount) {
    agent->printf("need money\n");
    return;
  }

  from_amount -= pay_amount;
  to_amount += pay_amount;

  from->cents = from_amount;
  to->cents = to_amount;

  agent->printf("ok\n");



  char buf[64];
  sprintf(buf, "paid by %s %lu", agent->who->nom.c_str(), pay_amount);
  string msg = buf;
  agent->server->notify(nom, msg);

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
  to->pushbrief(msg);
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
      agent->printf("%s\n", parson->tags[i]);
    }
  }
}

NEW_CMD(pass) {
  if (arg.size() > 1) {
    agent->printf("nope\n");
    return;
  }

  if (strchr(agent->who->nom.c_str(), '.')) {
    agent->printf("nope\n");
    return;
  }

  Parson *parson = agent->who->parson();
  if (!parson)
    return;

  string pass; 
  if (arg.size())
    pass = arg[0];

  if (pass.length() > 79) {
    agent->printf("nope\n");
    return;
  }

  parson->set_pass(pass);
  agent->printf("ok\n");
}

NEW_CMD(nomgen) {
  set<string> tags;
  for (auto tag : arg)
    tags.insert(tag);

  string nom;
  do {
    if (tags.count("female")) {
      nom = Parson::gen_nom(false);
    } else if (tags.count("male")) {
      nom = Parson::gen_nom(true);
    } else {
      nom = Parson::gen_nom();
    }
  } while (agent->server->urb->find(nom));
  assert(Parson::valid_nom(nom.c_str()));

  agent->printf("%s\n", nom.c_str());
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

NEW_CMD(restock) {
  if (arg.size() != 1)
    return;

  unsigned int n = strtoul(arg[0].c_str(), NULL, 0);
  agent->server->urb->restock(n);
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

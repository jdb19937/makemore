#define __MAKEMORE_SERVER_CC__ 1
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <sys/fcntl.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <netinet/in.h>
#include <string.h>

#include <string>
#include <vector>
#include <list>

#include "strutils.hh"
#include "parson.hh"
#include "server.hh"
#include "ppm.hh"
#include "commands.hh"
#include "agent.hh"

namespace makemore {

std::map<std::string, Server::Handler> Server::default_cmdtab;

using namespace std;

void Server::nonblock(int fd) {
  int flags = fcntl(fd, F_GETFL, 0);
  assert(flags > 0);
  int ret = fcntl(fd, F_SETFL, flags | O_NONBLOCK);
  assert(ret == 0);
}

static inline double realtime() {
  clock_t c = clock();
  return ((double)c / (double)CLOCKS_PER_SEC);
}

Server::Server(const std::string &_urbdir) {
  urbdir = _urbdir;
  port = 0;
  s = -1;
  urb = NULL;

  cmdtab = default_cmdtab;
}

Server::~Server() {
  kill();
  close();
}

void Server::open() {
  int ret;

  this->close();
  s = socket(PF_INET, SOCK_STREAM, 0);
  assert(s >= 0);

  int reuse = 1;
  ret = setsockopt(s, SOL_SOCKET, SO_REUSEPORT, (const char*)&reuse, sizeof(reuse));
  assert(ret == 0);
  ret = setsockopt(s, SOL_SOCKET, SO_REUSEADDR, (const char*)&reuse, sizeof(reuse));
  assert(ret == 0);

  nonblock(s);
}

void Server::close() {
  if (s >= 0) {
    ::close(s);
    port = 0;
    s = -1;
  }
}

void Server::bind(uint16_t _port) {
  assert(s >= 0);
  int ret;
  struct sockaddr_in sin;

  sin.sin_family = AF_INET;
  sin.sin_port = htons(_port);
  sin.sin_addr.s_addr = htonl(INADDR_ANY);
  ret = ::bind(s, (struct sockaddr *)&sin, sizeof(sin));
  assert(ret == 0);

  port = _port;
}

void Server::listen(int backlog) {
  assert(s >= 0);
  int ret = ::listen(s, backlog);
  assert(ret == 0);
}

void Server::setup() {
  fprintf(stderr, "opening urb %s\n", urbdir.c_str());
  urb = new Urb(urbdir.c_str(), 2);
  fprintf(stderr, "opened urb %s\n", urbdir.c_str());

  seedrand();
}


void Server::start(unsigned int kids) {
assert(0);

#if 0
  for (unsigned int i = 0; i < kids; ++i) {
    fprintf(stderr, "forking\n");
    if (pid_t pid = fork()) {
      pids.push_back(pid);
      fprintf(stderr, "forked pid=%d\n", pid);
      continue;
    }

    this->setup();
    while (1) {
      this->accept();
    }

    assert(0);
  }
#endif
}

static std::string ipstr(uint32_t ip) {
  char buf[INET_ADDRSTRLEN];
  const char *retbuf = inet_ntop(AF_INET, &ip, buf, INET_ADDRSTRLEN);
  assert(retbuf == buf);
  return std::string(buf);
}

void Server::accept() {
  fprintf(stderr, "accepting\n");

  for (unsigned int i = 0; i < 16; ++i) {
    struct sockaddr_in cin;
    socklen_t cinlen = sizeof(cin);
    int c = ::accept(s, (struct sockaddr *)&cin, &cinlen);
    if (c < 0)
      return;

    nonblock(c);

    uint32_t agent_ip;
    assert(sizeof(struct in_addr) == 4);
    memcpy(&agent_ip, &cin.sin_addr, 4);
    fprintf(stderr, "accepted %s\n", ipstr(agent_ip).c_str());

    Agent *agent = new Agent(this, NULL, c, agent_ip);
    assert(agent->who);

    nom_agent.insert(make_pair(agent->who->nom, agent));
    agents.insert(agent);
  }
}

void Server::renom(Agent *agent, const std::string &newnom) {
  assert(agent->who);
  if (agent->who->nom == newnom)
    return;

  auto naip = nom_agent.equal_range(agent->who->nom);
  for (auto nai = naip.first; nai != naip.second; ++nai) {
    if (nai->second == agent) {
      nom_agent.erase(nai);
      break;
    }
  }

  agent->who->become(newnom);

  nom_agent.insert(make_pair(agent->who->nom, agent));
}

void Server::notify(const std::string &nom, const std::string &msg, const Agent *exclude) {
  auto naip = nom_agent.equal_range(nom);

  for (auto nai = naip.first; nai != naip.second; ++nai) {
    const std::string &nom = nai->first;
    Agent *agent = nai->second;

    if (agent != exclude)
      agent->write(msg + "\n");
  }
}


void Server::wait() {
  for (auto pidi = pids.begin(); pidi != pids.end(); ++pidi) {
    pid_t pid = *pidi;
    fprintf(stderr, "waiting for pid=%d\n", pid);
    assert(pid == ::waitpid(pid, NULL, 0));
    fprintf(stderr, "waited\n");
  }
  pids.clear();
}

void Server::kill() {
  for (auto pidi = pids.begin(); pidi != pids.end(); ++pidi) {
    pid_t pid = *pidi;
    fprintf(stderr, "killing pid=%d\n", pid);
    ::kill(pid, 9);
    fprintf(stderr, "killed\n");
  }
}

void Server::websockify(uint16_t ws_port, const char *keydir) {
  assert(port > 0);
  pid_t ws_pid;

  ws_pid = fork();
  assert(ws_pid != -1);
  if (ws_pid) {
    pids.push_back(ws_pid);
    return;
  }

  char buf[256];
  sprintf(buf, ":%hu", port);
  std::string portstr = buf;
  sprintf(buf, "%hu", ws_port);
  std::string ws_portstr = buf;

  if (keydir) {
    string certstr = string(keydir) + string("/cert.pem");
    string privkeystr = string(keydir) + string("/privkey.pem");
    execlp("websockify",
      "websockify",
      "--cert", certstr.c_str(),
      "--key", privkeystr.c_str(),
      ws_portstr.c_str(), portstr.c_str(),
      NULL
    );
  } else {
    execlp("websockify",
      "websockify", ws_portstr.c_str(), portstr.c_str(),
      NULL
    );
  }

  assert(0);
}


void Server::select() {
  FD_ZERO(fdsets + 0);
  FD_ZERO(fdsets + 1);
  FD_ZERO(fdsets + 2);

  FD_SET(s, fdsets + 0);
  int nfds = s + 1;

  for (auto agent : agents) {
    if (agent->s >= nfds)
      nfds = agent->s + 1;

    if (agent->outbufn)
      FD_SET(agent->s, fdsets + 1);

    FD_SET(agent->s, fdsets + 0);
  }

  struct timeval timeout;
  timeout.tv_sec = 0;
  timeout.tv_usec = 50000;

//  fprintf(stderr, "calling select\n");
  int ret = ::select(nfds, fdsets + 0, fdsets + 1, fdsets + 2, &timeout);
//  fprintf(stderr, "select ret=%d\n", ret);
}

void Server::main() {
  while (1) {
    auto ai = agents.begin();
    while (ai != agents.end()) {
      Agent *agent = *ai;
      if (agent->s < 0) {
        fprintf(stderr, "closing %s\n", ipstr(agent->ip).c_str());

        auto naip = nom_agent.equal_range(agent->who->nom);
        for (auto nai = naip.first; nai != naip.second; ++nai) {
          if (nai->second == agent) {
            nom_agent.erase(nai);
            break;
          }
        }

        agents.erase(ai++);
        delete agent;
        continue;
      }
      ++ai;
    }
 

    this->select();

    if (FD_ISSET(s, fdsets + 0))
      this->accept();

    ai = agents.begin();
    while (ai != agents.end()) {
      Agent *agent = *ai;

      if (FD_ISSET(agent->s, fdsets + 0)) {
        if (!agent->slurp()) {
          fprintf(stderr, "closing %s\n", ipstr(agent->ip).c_str());

          auto naip = nom_agent.equal_range(agent->who->nom);
          for (auto nai = naip.first; nai != naip.second; ++nai) {
            if (nai->second == agent) {
              nom_agent.erase(nai);
              break;
            }
          }

          agents.erase(ai++);
          delete agent;
          continue;
        }
      }

      ++ai;
    }

    for (auto agent : agents) {
      if (FD_ISSET(agent->s, fdsets + 1)) {
        agent->flush();
      }
    }

    for (unsigned int i = 0; i < 64; ++i) {
      Parson *parson = urb->zones[0]->pick();
      if (!parson)
         continue;
      char *reqstr = parson->popbrief();
      if (!reqstr)
         continue;

      string req(reqstr);
      memset(reqstr, 0, Parson::briefsize);

      vector<string> reqwords;
      splitwords(req, &reqwords);

      Agent agent(this, parson->nom, -1, 0x0100007F);
      agent.command(reqwords);

      if (!agent.outbufn)
        continue;
      string rspstr = string(agent.outbuf, agent.outbufn);
fprintf(stderr, "rspstr=%s\n", rspstr.c_str());
      vector<string> rsps;
      splitlines(rspstr, &rsps);

      for (auto rspi = rsps.rbegin(); rspi != rsps.rend(); ++rspi) {
        parson->pushbrief(*rspi + " | " + req);
      }
    }


    for (auto agent : agents) {
      vector<vector<string > > lines;
      agent->parse(&lines);

      for (auto line : lines) {
        agent->command(line);
      }
    }
  }

}

}

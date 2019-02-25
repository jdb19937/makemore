#define __MAKEMORE_SERVER_CC__ 1
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
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

namespace makemore {

std::map<std::string, Server::Handler> Server::default_cmdtab;

using namespace std;

#define ensure(x) do { \
  if (!(x)) { \
    fprintf(stderr, "closing (%s)\n", #x); \
    goto done; \
  } \
} while (0)

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
}

static std::string ipstr(uint32_t ip) {
  char buf[INET_ADDRSTRLEN];
  const char *retbuf = inet_ntop(AF_INET, &ip, buf, INET_ADDRSTRLEN);
  assert(retbuf == buf);
  return std::string(buf);
}

void Server::accept() {
  stack.clear();

  fprintf(stderr, "accepting\n");

  struct sockaddr_in cin;
  socklen_t cinlen = sizeof(cin);
  int c = ::accept(s, (struct sockaddr *)&cin, &cinlen);
  assert(c != -1);

  assert(sizeof(struct in_addr) == 4);
  memcpy(&client_ip, &cin.sin_addr, 4);
  fprintf(stderr, "accepted %s\n", ipstr(client_ip).c_str());

  int c2 = ::dup(c);
  assert(c2 != -1);

  FILE *infp = ::fdopen(c, "rb");
  FILE *outfp = ::fdopen(c2, "wb");

  fprintf(stderr, "handling\n");
  this->handle(infp, outfp);
  fprintf(stderr, "handled\n");

  fclose(infp);
  fclose(outfp);
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

void Server::handle(FILE *infp, FILE *outfp) {
  Urbite self(ipstr(client_ip), urb);
  // fprintf(outfp, "hello %s\n", self.nom.c_str());

  while (1) {
    string line;
    if (!read_line(infp, &line))
      return;
    if (line == "")
      return;
    {
      size_t cr = line.find('\r');
      if (cr != string::npos)
        line.erase(cr);
    }

    vector<string> words;
    split(line.c_str(), ' ', &words);
    if (!words.size())
      return;
    string cmd = words[0];
    if (cmd == "")
      return;
    vector<string> args(&words.data()[1], &words.data()[words.size()]);

    fprintf(stderr, "got cmd %s\n", cmd.c_str());

    auto hi = cmdtab.find(cmd);
    if (hi == cmdtab.end())
      return;
    Handler h = hi->second;
    if (!h)
      return;
    if (!h((const Server *)this, urb, &self, cmd, args, infp, outfp))
      return;
    fflush(outfp);
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

void Server::think() {
  pid_t think_pid;

  think_pid = fork();
  assert(think_pid != -1);
  if (think_pid) {
    pids.push_back(think_pid);
    return;
  }

  setup();
  assert(urb);

  while (1) {
    fprintf(stderr, "thinking\n");
    sleep(1);
  }

  assert(0);
}

void Server::parcess(Parson *parson) {
  const unsigned int max_iters = 64;

  assert(urb);
  if (!parson->nom[0])
    return;

  list<string> cmds;
  {
    char *cmdstr;
    unsigned int cmdlen;
    while ((cmdstr = parson->popbuf(&cmdlen))) {
      string cmd(cmdstr);
      cmds.push_back(cmd);
      memset(cmdstr, 0, cmdlen);
    }
  }

  unsigned int iters = 0;
  auto cmdi = cmds.begin();
  while (cmdi != cmds.end()) {
    if (iters >= max_iters)
      break;
    ++iters;

    string cmd = *cmdi;
    cmds.erase(cmdi++);

    // send cmd to brane -> reqs
    vector<string> reqs;

    for (auto reqi = reqs.begin(); reqi != reqs.end(); ++reqi) {
      // send reqs to server -> new cmds
      vector<string> rsps;

      for (auto rspi = rsps.rbegin(); rspi != rsps.rend(); ++rspi)
        cmds.push_front(*reqi + ", " + *rspi);
    }
  }
}

}

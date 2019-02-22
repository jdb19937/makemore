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

#include "strutils.hh"
#include "parson.hh"
#include "server.hh"
#include "ppm.hh"
#include "commands.hh"

namespace makemore {

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
  s = -1;
  urb = NULL;

  setcmd("echo", cmd_echo);
  setcmd("GET", cmd_GET);
}

Server::~Server() {
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
    s = -1;
  }
}

void Server::bind(uint16_t port) {
  assert(s >= 0);
  int ret;
  struct sockaddr_in sin;

  sin.sin_family = AF_INET;
  sin.sin_port = htons(port);
  sin.sin_addr.s_addr = htonl(INADDR_ANY);
  ret = ::bind(s, (struct sockaddr *)&sin, sizeof(sin));
  assert(ret == 0);
}

void Server::listen(int backlog) {
  assert(s >= 0);
  int ret = ::listen(s, backlog);
  assert(ret == 0);
}

void Server::setup() {
  fprintf(stderr, "opening urb %s\n", urbdir.c_str());
  urb = new Urb(urbdir.c_str());
  fprintf(stderr, "opened urb %s\n", urbdir.c_str());
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

void Server::accept() {
  fprintf(stderr, "accepting\n");

  struct sockaddr_in sin;
  socklen_t sinlen = sizeof(sin);
  int c = ::accept(s, (struct sockaddr *)&sin, &sinlen);
  assert(c != -1);

  fprintf(stderr, "accepted\n");

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
  while (1) {
    string line;
    if (!read_line(infp, &line))
      return;
    if (line == "")
      return;
    {
      size_t cr = line.find('\r');
      if (cr > 0)
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

    Handler h = cmdtab[cmd];
    if (!h)
      return;
    if (!h(this, cmd, args, infp, outfp))
      return;
    fflush(outfp);
  }
}
   

}

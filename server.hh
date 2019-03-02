#ifndef __MAKEMORE_SERVER_HH__
#define __MAKEMORE_SERVER_HH__ 1

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>

#include <vector>
#include <map>
#include <set>
#include <string>

#include "urb.hh"
#include "urbite.hh"
#include "strutils.hh"

namespace makemore {

struct Server {
  static void nonblock(int fd);

  typedef void (*Handler)(
    class Agent *,
    const std::vector<std::vector<std::string> > &ctx,
    const std::string &cmd,
    const std::vector<std::string> &arg
  );

  static std::map<std::string, Handler> default_cmdtab;

  static bool default_cmdset(const std::string &cmd, Handler h) {
    fprintf(stderr, "adding default server command %s\n", cmd.c_str());
    assert(cmd.length() < 32);
    default_cmdtab[cmd] = h;
    return true;
  }

  int s;
  uint16_t port;
  fd_set fdsets[3];

  std::set<class Agent*> agents;
  std::multimap<std::string, class Agent *> nom_agent;
  std::vector<pid_t> pids;
  std::map<std::string, Handler> cmdtab;

  std::string urbdir;
  Urb *urb;


  Server(const std::string &urb);
  ~Server();

  void open();
  void close();
  void bind(uint16_t _port);
  void listen(int backlog = 256);
  void select();

  void renom(Agent *agent, const std::string &nom);
  void notify(const std::string &nom, const std::string &msg, const Agent *exclude = NULL);

  void websockify(uint16_t ws_port, const char *keydir);

  void websockify(uint16_t ws_port) {
    websockify(ws_port, NULL);
  }
  void websockify(uint16_t ws_port, const std::string &keydir) {
    websockify(ws_port, keydir.c_str());
  }

  void start(unsigned int kids = 8);
  void accept();
  void wait();
  void kill();

  void setup();

  bool cmdset(const std::string &cmd, Handler h) {
    assert(cmd.length() < 32);
    cmdtab[cmd] = h;
    return true;
  }

  void think();

  void main();
};

}

#endif

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
#include <string>

#include "urb.hh"
#include "urbite.hh"

namespace makemore {


struct Server {
  typedef bool (*Handler)(
    const Server *,
    Urb *,
    Urbite *,
    const std::string &cmd,
    const std::vector<std::string> &args,
    FILE *infp,
    FILE *outfp)
  ;
  static std::map<std::string, Handler> default_cmdtab;

  static bool default_cmdset(const std::string &cmd, Handler h) {
    fprintf(stderr, "adding default server command %s\n", cmd.c_str());
    assert(cmd.length() < 32);
    default_cmdtab[cmd] = h;
    return true;
  }

  int s;
  uint16_t port;
  std::vector<pid_t> pids;
  std::map<std::string, Handler> cmdtab;

  std::string urbdir;
  Urb *urb;

  uint32_t client_ip;

  std::vector<Parson*> stack;

  Server(const std::string &urb);
  ~Server();

  void open();
  void close();
  void bind(uint16_t _port);
  void listen(int backlog = 256);

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
  void handle(FILE *infp, FILE *outfp);

  bool cmdset(const std::string &cmd, Handler h) {
    assert(cmd.length() < 32);
    cmdtab[cmd] = h;
    return true;
  }
};

}

#endif

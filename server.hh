#ifndef __MAKEMORE_SERVER_HH__
#define __MAKEMORE_SERVER_HH__ 1

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/types.h>

#include <vector>
#include <map>
#include <string>

#include "urb.hh"

namespace makemore {

struct Server {
  typedef bool (*Handler)(Server *, const std::string &, const std::vector<std::string> &, FILE *, FILE *);

  int s;
  std::vector<pid_t> pids;
  std::map<std::string, Handler> cmdtab;

  std::string urbdir;
  class Urb *urb;

  Server(const std::string &urb);
  ~Server();

  void open();
  void close();
  void bind(uint16_t port);
  void listen(int backlog = 256);

  void start(unsigned int kids = 8);
  void accept();
  void wait();
  void kill();

  void setup();
  void handle(FILE *infp, FILE *outfp);

  void setcmd(const std::string &cmd, Handler h) {
    assert(cmd.length() < 32);
    cmdtab[cmd] = h;
  }
};

}

#endif

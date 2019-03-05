#ifndef __MAKEMORE_AGENT_HH__
#define __MAKEMORE_AGENT_HH__ 1

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

struct Agent {
  class Server *server;
  int s;
  uint32_t ip;
  std::string ipstr;

  bool secure;
  SSL *ssl;
  int ssl_status;

  enum { UNKNOWN, HTTP, MORETP } proto;
  std::vector<std::string> httpbuf;
  bool httpkeep;
  std::string httpua;
  std::string httpvers;

  char *inbuf;
  unsigned int inbufm, inbufn;
  unsigned int inbufj, inbufk;

  char *outbuf;
  unsigned int outbufm, outbufn;

  Urbite *who;

  Agent(class Server *_server, const char *nom, int _s = -1, uint32_t _ip = 0x7F000001U, bool _secure = false);
  ~Agent();
  void close();

  bool slurp();
  void parse(std::vector< std::vector<std::string> > *lines);
  void write(const std::string &str);
  void printf(const char *fmt, ...);
  void flush();
  void command(const std::vector<std::string> &line);
};

}

#endif

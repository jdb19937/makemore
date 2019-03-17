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
#include "process.hh"
#include "strutils.hh"
#include "session.hh"

struct ssl_st;
typedef struct ssl_st SSL;

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
  std::string httppath;
  std::string httpvers;

  char *inbuf;
  unsigned int inbufm, inbufn;
  unsigned int inbufj, inbufk;
  strmat linebuf;
  bool need_slurp() {
    return (inbufn < inbufm && linebuf.empty());
  }

  char *outbuf;
  unsigned int outbufm, outbufn;
  bool need_flush() {
    return (
      outbufn > 0 ||
      (session->shell && session->shell->otab.size() && session->shell->otab[0] && session->shell->otab[0]->can_get())
    );
  }

  Session *session;
  Urbite *who() {
    assert(session);
    return session->who;
  }

  Agent(class Server *_server, int _s = -1, uint32_t _ip = 0x7F000001U, bool _secure = false);
  ~Agent();
  void close();

  bool slurp();
  void parse();
  bool write(const std::string &str);
  bool write(const strvec &vec);
  bool write(const Line &wvec);
  void printf(const char *fmt, ...);
  void flush();
  void command(const std::vector<std::string> &line);
};

}

#endif

#ifndef __MAKEMORE_SESSION_HH__
#define __MAKEMORE_SESSION_HH__ 1

#include <stdlib.h>

#include <map>
#include <string>

#include "strutils.hh"

namespace makemore {

struct Session {
  struct Server *server;
  struct Agent *agent;
  struct Urbite *who;

  struct Process *shell;

  struct Session *prev_reader, *next_reader;
  struct Session *prev_writer, *next_writer;
  struct Process *head_sproc;


  std::map<std::string, std::string> wordvar;
  std::map<std::string, strvec> linevar;
  std::map<std::string, strmat> gridvar;


  Session(Agent *_agent);
  ~Session();

  void link_sproc(Process *p);
  void unlink_sproc(Process *p);
};

}

#endif

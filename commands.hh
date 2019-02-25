#ifndef __MAKEMORE_COMMANDS_HH__
#define __MAKEMORE_COMMANDS_HH__ 1

#include <stdio.h>

#include <string>
#include <vector>

#include "server.hh"
#include "urbite.hh"

namespace makemore {

#define CMD_ARGS \
  const Server *_server, \
  Urb *urb, \
  Urbite *self, \
  const std::string &cmd, \
  const std::vector<std::string> &args, \
  FILE *infp, \
  FILE *outfp

struct _OnStartup {
  _OnStartup(bool x) { }
};

// stop compiler from optimizing away declarations
extern int _startup_count;
static int _ref_startup_count = _startup_count;

#define NEW_CMD(name) \
  bool cmd_ ## name ( CMD_ARGS ); \
  _OnStartup _add_cmd_ ## name (_startup_count += (int)Server::default_cmdset(#name, cmd_ ## name )); \
  bool cmd_ ## name ( CMD_ARGS ) 
  
}
#endif

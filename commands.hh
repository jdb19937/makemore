#ifndef __MAKEMORE_COMMANDS_HH__
#define __MAKEMORE_COMMANDS_HH__ 1

#include <stdio.h>

#include <string>
#include <vector>

#include "server.hh"

namespace makemore {

#define CMD_ARGS \
  class Server *server, \
  const std::string &cmd, \
  const std::vector<std::string> &args, \
  FILE *infp, \
  FILE *outfp
  
extern bool cmd_echo(CMD_ARGS);
extern bool cmd_GET(CMD_ARGS);

}
#endif

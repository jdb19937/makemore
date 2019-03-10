#ifndef __MAKEMORE_COMMAND_HH__
#define __MAKEMORE_COMMAND_HH__ 1

#include <string>
#include <vector>

namespace makemore {

typedef enum {
  COMMAND_STATE_INITIALIZE,
  COMMAND_STATE_RUNNING,
  COMMAND_STATE_SHUTDOWN
} CommandState;

typedef bool (*Command)(
  class Server *,
  class Urbite *,
  class Process *,
  CommandState state,
  const std::vector<std::string> &args
);

extern Command find_command(const std::string &cmd);

}

#endif
